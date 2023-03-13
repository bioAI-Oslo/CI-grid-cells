import torch

from methods import HexagonalGCs


class Similitude(HexagonalGCs):
    def __init__(self, **kwargs):
        super(Similitude, self).__init__(**kwargs)

    def loss_fn(self, r):
        J = self.jacobian(r)
        det_J = self.the_jacobian(J)
        return torch.var(det_J)
    
class PlaceCells(HexagonalGCs):
    """
    torch model for learning optimal place cell phases
    """

    def __init__(self, ncells=3, f = 1, init_rot = 0, sig=1, scale = None, dtype=torch.float32, **kwargs):
        super(PlaceCells, self).__init__(ncells, f, init_rot, **kwargs)
        
        self.sig = sig # place cell tuning width 
        # conformal scaling factor.
        if scale is None:
            self.scale = torch.nn.Parameter(torch.ones(1,dtype = torch.float32))
        else:
            self.scale = scale
        

    def forward(self, r, rectify=False):
        """
        Parameters:
            r (nsamples,2): spatial samples
        Returns:
            activity (nsamples,ncells): activity of all cells on spatial samples
        """
        activity = torch.exp(-torch.sum((r[:, None] - self.phases[None])**2/(2*self.sig**2), dim = -1))
        return activity

    def jacobian(self, r):
        """
        Jacobian of the forward function

        Parameters:
            r (nsamples,2): spatial samples
        Returns:
            J (nsamples,ncells,2): jacobian of the forward function
        """
        p = self(r)
        J = -1/self.sig**2*(r[:,None] - self.phases[None])*p[...,None]   
        return J

    def metric(self, r):
        J = self.jacobian(r)
        return torch.transpose(J, -2, -1) @ J
    
    def loss_fn(self, r):
        """
        Scaled conformal Isometry loss
        """
        g = self.metric(r)        
        diag_loss = (g[:,0,0] - self.scale*100)**2 +  (g[:,1,1] - self.scale*100)**2 
        cross_loss = 2*g[:,0,1]**2
        return torch.mean(diag_loss+cross_loss)
    


class Similitude2(HexagonalGCs):
    def __init__(self, scale=None, **kwargs):
        super(Similitude2, self).__init__(**kwargs)
        # scale of similitude
        self.set_scale(scale)
        self.optimizer = torch.optim.Adam(self.parameters(),lr=0.001)

    def set_scale(self, scale=None):
        if scale is None:
            # conformally isometric scaling LAW 
            scale = self.ncells*0.014621597785714284
        self.scale = torch.nn.Parameter(torch.tensor(scale,dtype=self.dtype),requires_grad=True)
        return self.scale

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
        return torch.mean(torch.log(loss))
