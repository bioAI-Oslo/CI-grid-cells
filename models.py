from methods import *


class Similitude(HexagonalGCs):
    def __init__(self,**kwargs):
        super(Similitude, self).__init__(**kwargs)
    
    def loss_fn(self,r):
        J = self.jacobian(r)
        det_J = self.the_jacobian(J)
        return torch.var(det_J)