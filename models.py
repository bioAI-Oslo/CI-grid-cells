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
        self.optimizer = torch.optim.Adam(self.parameters(),lr=0.001)
        self.set_scale(scale)

    def set_scale(self, scale=None):
        if scale is None:
            # conformally isometric scaling LAW 
            scale = self.ncells*0.014621597785714284
        self.scale = torch.nn.Parameter(torch.tensor(scale,dtype=torch.float32),requires_grad=True)
        return self.scale

    def loss_fn(self, r):
        J = self.jacobian(r)
        # (nsamples,2,2)
        metric_tensor = self.metric_tensor(J)
        diag_elems = torch.diagonal(metric_tensor, dim1=-2, dim2=-1)
        lower_triangular_elems = torch.tril(metric_tensor, diagonal=-1)
        loss = torch.sum((diag_elems - 100*self.scale) ** 2, dim=-1) + 2 * torch.sum(
            lower_triangular_elems**2,dim=(-2, -1)
        )
        # HOW TO INCLUDE MAXIMISE SCALE????
        #loss -= 100*self.scale

        # g11 = metric_tensor[...,0,0]
        # g22 = metric_tensor[...,1,1]
        # off_diag = metric_tensor[...,1,0]
        # det = g11*g22 - off_diag**2
        # loss = (det - self.scale)**2
        return torch.mean(loss)

class Similitude3(HexagonalGCs):
    def __init__(self, scale=None, shift = -1, **kwargs):
        super(Similitude3, self).__init__(**kwargs)
        # scale of similitude
        if scale is None:
            # conformally isometric scaling LAW 
            scale = self.ncells*0.014621597785714284
        self.scale = torch.nn.Parameter(torch.tensor(scale,dtype=torch.float32), requires_grad = True)
        self.optimizer = torch.optim.Adam(self.parameters(),lr=0.001)
        self.shift = shift
        
    def forward(self, r):
        jittered_phases = self.phases + r[-1]
        activity = torch.cos((r[0][:, None] - jittered_phases[None]) @ self.ks.T)
        activity = torch.sum(activity, dim=-1)  # sum plane waves         
        activity =  2/9*(1-self.shift)*activity + 1/3*(1 + 2*self.shift)
        activity = self.relu(activity) if self.relu else activity
        return activity 
    
    def jacobian(self, r):
        """
        Jacobian of the forward function
        """
        jittered_phases = self.phases + r[-1]
        J_tmp = -2/9*(1-self.shift)*torch.sin((r[0][:, None] - jittered_phases[None]) @ self.ks.T)

        Jx = torch.sum(J_tmp * self.ks[:, 0], dim=-1)
        Jy = torch.sum(J_tmp * self.ks[:, 1], dim=-1)
        
        J = torch.stack([Jx, Jy], dim=-1)
        if self.relu:
            relu_grad_mask = self.forward(r) > 0
            J = relu_grad_mask[..., None] * J
        return J
    
    def metric_metrics(self, r, keepdims = False):
        J = self.jacobian(r)
        metric_tensor = self.metric_tensor(J)
        
        g11 = metric_tensor[...,0,0]
        g22 = metric_tensor[...,1,1]
        
        off_diag = metric_tensor[...,1,0]
        return g11, g22, off_diag 

    def loss_fn(self, r, keepdims = False):
        ra = r[0]
        rb = ra + r[1]
        rc = ra + r[2] 
        dr = r[3]
                
        ga = self((ra, r[-1]))
        gb = self((rb, r[-1]))
        gc = self((rc, r[-1])) 
        
        sab = torch.sum((ga - gb)**2,dim = -1)/dr**2
        sac = torch.sum((ga - gc)**2,dim = -1)/dr**2
        
        loss = (sab - sac)**2
        
        if not keepdims:
            return torch.mean(loss)
        else: 
            return loss
    
    def metric_loss(self, r, keepdims = False):
        g11, g22, off_diag = self.metric_metrics((r[0], r[-1]))
        
        loss = torch.var(g11) + torch.var(g22) + torch.mean((g11 - g22)**2) + 2*torch.mean(off_diag**2)
        
        if not keepdims:
            return torch.mean(loss)
        else: 
            return loss