import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from importlib import import_module
from core.anchors import generate_default_anchor_maps, hard_nms
from torch.autograd import Variable
from .grad_cam import GradCam
from torch.nn.modules.module import Module
import random

criterion = nn.CrossEntropyLoss()
def l2_norm_v2(input):
    input_size = input.size()
    _output = input/(torch.norm(input, p=2, dim=-1, keepdim=True))
    output = _output.view(input_size)
    return output

class Classifier(nn.Module):
    def __init__(self, in_panel, out_panel, bias=False):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_panel, out_panel, bias=bias)

    def forward(self, input):
        logit = self.fc(input)
        if logit.dim()==1:
            logit =logit.unsqueeze(0)
        return logit



def get_bbox(x, conv5, layer_weights, rate=0.2):
    conv5_cam = conv5.clone().detach().view(conv5.size(0), conv5.size(1), 14*14) * layer_weights.unsqueeze(-1)
    mask_sum = conv5_cam.sum(1, keepdim=True).view(conv5.size(0), 1, 14, 14)
    mask_sum = F.interpolate(mask_sum, size=(448, 448), mode='bilinear', align_corners=True)
    mask_sum = mask_sum.view(mask_sum.size(0), -1)
    x_range = mask_sum.max(-1, keepdim=True)[0] - mask_sum.min(-1, keepdim=True)[0]
    mask_sum = (mask_sum - mask_sum.min(-1, keepdim=True)[0])/x_range
    mask = torch.sign(torch.sign(mask_sum - rate) + 1)
    mask = mask.view(mask.size(0), 1, 448, 448)
    input_box = torch.zeros_like(x)
    xy_list = []
    for k in torch.arange(x.size(0)):
        indices = mask[k].nonzero()
        y1, x1 = indices.min(dim=0)[0][-2:]
        y2, x2 = indices.max(dim=0)[0][-2:]
        tmp = x[k, :, y1:y2, x1:x2]
        if x1==x2 or y1==y2:
            tmp = x[k, :, :, :]
        input_box[k] = F.interpolate(tmp.unsqueeze(0), size=(448, 448), mode='bilinear', align_corners=True).clone().detach().cuda()
        xy_list.append([x1, x2, y1, y2])
    return input_box, xy_list


def Mask(nb_batch, channels):

    foo = [1] * 3 + [0] *  1
    bar = []
    for i in range(200):
        random.shuffle(foo)
        bar += foo

    bar = [bar for i in range(nb_batch)]
    bar = np.array(bar).astype("float32")
    bar = bar.reshape(nb_batch, 200*channels, 1, 1)
    bar = torch.from_numpy(bar)
    bar = bar.cuda()
    bar = Variable(bar)
    return bar

class my_MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(my_MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        input = input.transpose(3,1)
        # transpose will break contiguous, annotated by lm

        input = F.max_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)
        input = input.transpose(3,1).contiguous()

        return input


class ProposalNet(nn.Module):
    def __init__(self):
        super(ProposalNet, self).__init__()
        self.down1 = nn.Conv2d(2048, 128, 3, 1, 1)
        self.down2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.down3 = nn.Conv2d(128, 128, 3, 2, 1)
        self.ReLU = nn.ReLU()
        self.tidy1 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy2 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy3 = nn.Conv2d(128, 9, 1, 1, 0)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.downsample1 = nn.Conv2d(128, 128, 3, 2, 1)
        self.downsample2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def unsample1(self, x, y):
        _, _, H, W = x.size()
        t = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)
        t = torch.mean(t, dim=1, keepdim=True)
        t = self.sigmoid(t)
        return t
    def forward(self, x):
        batch_size = x.size(0)
        d1 = self.ReLU(self.down1(x))
        d2 = self.ReLU(self.down2(d1))
        d3 = self.ReLU(self.down3(d2))

        d2_1 = self.unsample1(d1, d2)
        e2_1 = d2_1 * d1
        d1_final = d1 - e2_1
        d2_2 = d2 + self.downsample1(e2_1)

        d3_1 = self.unsample1(d2_2, d3)
        e3_1 = d3_1 * d2_2
        d2_final = d2_2 - e3_1
        d3_final = d3 + self.downsample1(e3_1)

        t1 = self.tidy1(d1_final).view(batch_size, -1)
        t2 = self.tidy2(d2_final).view(batch_size, -1)
        t3 = self.tidy3(d3_final).view(batch_size, -1)
        return torch.cat((t1, t2, t3), dim=1)

class PSPModule(nn.Module):
    def __init__(self, sizes=(1, 2, 3), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center
class SearchTransfer(nn.Module):
    def __init__(self):
        super(SearchTransfer, self).__init__()
        self.conv_trans = nn.Conv2d(4096, 2048, 1, 1, 0)
        self.conv_q = nn.Conv2d(2048, 2048, 1, 1, 0)
        self.conv_k = nn.Conv2d(2048, 2048, 1, 1, 0)
        self.conv_v = nn.Conv2d(2048, 2048, 1, 1, 0)
    def bis(self, input, dim, index):
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, part_ref,part_target):
        part_ref = self.conv_k(part_ref)
        part_target_o = part_target
        part_target = self.conv_q(part_target)
        part_target_unfold = F.unfold(part_target, kernel_size=(3, 3), padding=1)
        part_ref_unfold1 = F.unfold(part_ref, kernel_size=(3, 3), padding=1)
        part_ref_unfold = part_ref_unfold1.permute(0, 2, 1)

        part_ref_unfold = F.normalize(part_ref_unfold, dim=2) 
        part_target_unfold  = F.normalize(part_target_unfold, dim=1) 
        R_part = torch.bmm(part_ref_unfold, part_target_unfold) 
        R_part_star, R_part_arg = torch.max(R_part, dim=1) 
        T_unfold = self.bis(part_ref_unfold1, 2, R_part_arg)
        T_part = F.fold(T_unfold, output_size=part_ref.size()[-2:], kernel_size=(3,3), padding=1) / (3.*3.)
        S = R_part_star.view(R_part_star.size(0), 1, T_part.size(2), T_part.size(3))
        x_res = torch.cat((part_target, T_part), dim=1)
        part_res = self.conv_trans(x_res)
        part_res = part_res * S
        part_res = part_res + part_target_o
        return part_res

class ContextBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 psp_size = (1, 3, 5)
                 ):
        super(ContextBlock, self).__init__()
        self.psp = PSPModule(psp_size)
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)

        self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
            nn.LayerNorm([self.planes, 7, 7]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.planes, self.inplanes, kernel_size=1))


    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        input_x = input_x.view(batch, height * width, channel)
        value = self.psp(x)
        value = self.softmax(value)
        sim_map = torch.matmul(input_x, value)
        context_mask = torch.matmul(sim_map, value.view(batch, value.size(2), channel))
        context = context_mask.view(batch, channel, height, width)
        return context

    def forward(self, x):
        # [N, C, H, W]
        context = self.spatial_pool(x)
        out = x
        channel_add_term = self.channel_add_conv(context)
        out = out + channel_add_term
        return out

class CPNet(nn.Module):
    def __init__(self, opt=None):
        super(CPNet, self).__init__()
        num_classes = opt.num_classes
        # ----main branch----
        topN = 4
        basenet = getattr(import_module('torchvision.models'), opt.arch)
        basenet = basenet(pretrained=True)

        self.conv4 = nn.Sequential(*list(basenet.children())[:-3])
        self.conv5 = nn.Sequential(*list(basenet.children())[-3])
        self.conv6 = nn.Conv2d(1024, 10 * num_classes, 1, 1, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = Classifier(2048, num_classes, bias=True)

        # ----other branch----
        self.cls_cat = Classifier(2048+10*num_classes, num_classes, bias=True)
        self.cls_part = Classifier(10 * num_classes, num_classes, bias=True)

        self.cls_cat_a = Classifier(3*(2048+10*num_classes), num_classes, bias=True)
        self.num_classes = num_classes
        self.box_thred = opt.box_thred
        self.feat_extractor = nn.Sequential(*list(basenet.children())[:-2])
        
        self.conv_fc = nn.Linear(512 * 4, 200)
        self.proposal_net = ProposalNet()
        self.topN = topN
        self.global_context = ContextBlock(2048, 4)
        #

        self.SearchTransfer1 = SearchTransfer()
        self.SearchTransfer2 = SearchTransfer()
        self.SearchTransfer3 = SearchTransfer()

        self.concat_net = nn.Linear(2048 * (4 + 1), 200)
        self.cls_out = nn.Linear(2048 * 4, 200)
        self.part_cls = nn.Linear(512 * 4, 200)
        self.cls_trans = Classifier(512 * 4, num_classes, bias=True)
        # anchors generating
        _, edge_anchors, _ = generate_default_anchor_maps()
        self.pad_side = 224
        self.edge_anchors = (edge_anchors + 224).astype(np.int)

        

    def forward(self, x, y=None, is_vis=False, vis_idx=None, gt_top=None):
        # ---raw image: x
        batch = x.size(0)
        conv4 = self.conv4(x)
        conv5 = self.conv5(conv4)
        conv5_pool = self.avg_pool(conv5).view(batch, -1)
        pool_conv6 = self.max_pool(F.relu(self.conv6(conv4.detach()))).view(batch, -1)

        pool_cat = torch.cat((10*l2_norm_v2(conv5_pool.detach()), 10*l2_norm_v2(pool_conv6.detach())), dim=1)
        raw_logits = self.cls_cat(pool_cat)

        # ---central image: central_image
        with torch.enable_grad():
            layer_weights = None
            self.grad_cam = GradCam(model = self,
                                    feature_extractor=self.conv5,
                                    classifier=self.classifier,
                                    target_layer_names = ["2"],
                                    use_cuda=True)
            target_index = y
            if is_vis and (not vis_idx is None):
                if vis_idx==1:
                    target_index=gt_top
            layer_weights = self.grad_cam(conv4.detach(), target_index)
        central_image, _ = get_bbox(x, conv5, layer_weights, rate=self.box_thred)
        central_feature =self.feat_extractor(central_image)
        central_vec = nn.Dropout(p=0.)(self.avg_pool(central_feature).view(batch,-1))
        central_logits = self.conv_fc(central_vec)

        # ---part images: part_imgs
        x_pad = F.pad(central_image, (self.pad_side, self.pad_side, self.pad_side, self.pad_side), mode='constant', value=0)
        rpn_score = self.proposal_net(central_feature.detach())
        all_cdds = [
            np.concatenate((x.reshape(-1, 1), self.edge_anchors.copy(), np.arange(0, len(x)).reshape(-1, 1)), axis=1)
            for x in rpn_score.data.cpu().numpy()]
        top_n_cdds = [hard_nms(x, topn=self.topN, iou_thresh=0.20) for x in all_cdds]
        top_n_cdds = np.array(top_n_cdds)
        top_n_index = top_n_cdds[:, :, -1].astype(np.int)
        top_n_index = torch.from_numpy(top_n_index).cuda()

        top_n_index = torch.tensor(top_n_index, dtype=torch.long).cuda()
        part_imgs = torch.zeros([batch, self.topN, 3, 224, 224]).cuda()

        for i in range(batch):
            for j in range(self.topN):
                [y0, x0, y1, x1] = top_n_cdds[i][j, 1:5].astype(np.int)
                part_imgs[i:i + 1, j] = F.interpolate(x_pad[i:i + 1, :, y0:y1, x0:x1], size=(224, 224), mode='bilinear',
                                                      align_corners=True)
        part_imgs = part_imgs.view(batch * self.topN, 3, 224, 224)
        part_features_all= self.feat_extractor(part_imgs.detach())

        part_features_all = part_features_all.view(batch, self.topN, -1, 7, 7)
        part_features_I0 = part_features_all[:, 0, ...]
        part_features_I1 = part_features_all[:, 1, ...]
        part_features_I2 = part_features_all[:, 2, ...]
        part_features_I3 = part_features_all[:, 3, ...]
        S1 = self.SearchTransfer1(part_features_I0, part_features_I1)
        S2 = self.SearchTransfer2(part_features_I0, part_features_I2)
        S3 = self.SearchTransfer3(part_features_I0, part_features_I3)
        part_features_tran = torch.cat((part_features_I0,S1,S2,S3),)
        out_features = self.avg_pool(self.global_context(part_features_tran))
        out_vec = out_features.view(batch, -1)
        out_logits = self.cls_out(out_vec)
        logits_gate = torch.stack([raw_logits,central_logits,out_logits], dim=-1)
        logits_gate = logits_gate.sum(-1)

        if not self.training:
            return logits_gate
