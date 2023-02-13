import torch

from methods import HexagonalGCs


class Similitude(HexagonalGCs):
    def __init__(self, **kwargs):
        super(Similitude, self).__init__(**kwargs)

    def loss_fn(self, r):
        J = self.jacobian(r)
        det_J = self.the_jacobian(J)
        return torch.var(det_J)


class Similitude2(HexagonalGCs):
    def __init__(self, scale=None, **kwargs):
        super(Similitude2, self).__init__(**kwargs)
        # scale of similitude
        if scale is None:
            # conformally isometric scaling LAW 
            scale = self.ncells*0.014621597785714284
        self.scale = torch.nn.Parameter(torch.tensor(scale,dtype=torch.float32),requires_grad=True)
        self.optimizer = torch.optim.Adam(self.parameters(),lr=0.001)

    def loss_fn(self, r):
        J = self.jacobian(r)
        # (nsamples,2,2)
        metric_tensor = self.metric_tensor(J)
        diag_elems = torch.diagonal(metric_tensor, dim1=-2, dim2=-1)
        lower_triangular_elems = torch.tril(metric_tensor, diagonal=-1)
        loss = torch.sum((diag_elems - 100*self.scale) ** 2, axis=-1) + 2 * torch.sum(
            lower_triangular_elems**2, axis=(-2, -1)
        )
        # HOW TO INCLUDE MAXIMISE SCALE????
        #loss -= 100*self.scale

        # g11 = metric_tensor[...,0,0]
        # g22 = metric_tensor[...,1,1]
        # off_diag = metric_tensor[...,1,0]
        # det = g11*g22 - off_diag**2
        # loss = (det - self.scale)**2
        return torch.mean(loss)
