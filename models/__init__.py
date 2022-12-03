from .cp import CPNet
def get_model(opt):
    
    net = CPNet(opt)
    return net

