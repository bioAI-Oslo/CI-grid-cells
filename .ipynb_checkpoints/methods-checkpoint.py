import numpy as np
import torch
import copy

from utils import *


class Hexagon:
    def __init__(self, radius, orientation_offset, center):
        self.radius = radius
        self.orientation_offset = orientation_offset
        self.center = center
        self.area = 3 * np.sqrt(3) * radius * radius / 2
        # create hexagonal points
        rotmat60 = rotation_matrix(60, degrees=True)
        rotmat_offset = rotation_matrix(orientation_offset, degrees=True)
        hpoints = np.array([radius, 0])  # start vector along cardinal x-axis
        hpoints = rotmat_offset @ hpoints
        hpoints = [hpoints]
        for _ in range(5):
            hpoints.append(rotmat60 @ hpoints[-1])
        self.hpoints = np.array(hpoints)
        self.basis = np.zeros((6, 2))
        for i in range(6):
            self.basis[i] = (hpoints[i] + hpoints[(i + 1) % 6]) / 2

    def is_in_hexagon(self, point):
        """
        Check if a 2d-point is within a hexagon defined by its 6
        points 'hpoints' with phase 'center'.
        """
        u2, v2 = self.center, point - self.center
        hpoints = self.hpoints + self.center
        for i in range(6):
            # loop each hexagonal side/edge
            u1 = hpoints[i]
            v1 = hpoints[(i + 1) % 6] - u1
            _, intersect_inside_hexagon = intersect(
                u1, v1, u2, v2, constraint1=[0, 1], constraint2=[0, 1]
            )
            if intersect_inside_hexagon:
                return False
        return True

    def sample(self, N, seed=None):
        # sample points within hexagon
        rng = np.random.default_rng(seed)
        samples = np.zeros((N, 2))
        for i in range(N):
            sample_square = rng.uniform(-self.radius, self.radius, 2)
            while not self.is_in_hexagon(sample_square):
                sample_square = rng.uniform(-self.radius, self.radius, 2)
            samples[i] = sample_square
        return samples

    def wrap(self, xs, hexagon=None):
        """
        Loops nsamples of x to the _wrap function.
        
        Parameters:
            x (nsamples, 2): sequence of 2D arrays giving 2D positions.
        """
        return np.array([self._wrap(x) for x in xs])
    
    def _wrap(self, x, hexagon=None):
        """
        Simple wrapping method that draws hexagons surrounding the
        vector x. The final hexagon containing the end point of the vector x
        gives the wrapped location of x as: x - origin
        
        Parameters:
            x (2,): 2D np.ndarray giving a position in 2D space
            hexagon: object of this class - should not be specified by user, but
                     it is used by this method during recursion.
        """
        hexagon = copy.deepcopy(self) if hexagon is None else hexagon
        if hexagon.is_in_hexagon(x):
            return x - hexagon.center
        hexdrant = np.argmax(hexagon.basis @ (x - hexagon.center))
        hexagon.center += 2*hexagon.basis[hexdrant]
        return self._wrap(x, hexagon)
    
    def geodesic(self,p1,p2):
        """
        Parameters:
            p1,p2 (nsamples,2): nsamples of 2D vectors
        """
        p1 = self.wrap(p1)
        p2 = self.wrap(p2)
        p2 = np.concatenate([p2[:,None], p2[:,None] - 2*self.basis[None]], axis=1) # (nsamples,7,2)
        dist = np.linalg.norm(p2-p1[:,None],axis=-1) # (nsamples,7)
        return np.min(dist,axis=-1)

    def plot(self, fig, ax, center=None, color="blue"):
        center = self.center if center is None else center
        hpoints = self.hpoints + center
        for i in range(len(hpoints)):
            ax.plot(*hpoints[i : (i + 2)].T, color=color)
        last_line = np.array([hpoints[-1], hpoints[0]])
        ax.plot(*last_line.T, color=color)
        ax.set_aspect("equal")
        return fig, ax


class HexagonalGCs(torch.nn.Module):
    """
    torch model for learning optimal grid cell phases
    """

    def __init__(self, ncells=3, f=1, init_rot=0, dtype=torch.float32, **kwargs):
        super(HexagonalGCs, self).__init__(**kwargs)
        # init static grid properties
        self.ncells, self.f, self.init_rot, self.dtype = ncells, f, init_rot, dtype
        rotmat_init = rotation_matrix(init_rot)
        rotmat_60 = rotation_matrix(60)
        k1 = np.array([1.0, 0.0])
        k1 = rotmat_init @ k1
        ks = np.array([np.linalg.matrix_power(rotmat_60, k) @ k1 for k in range(3)])
        ks = torch.tensor(ks, dtype=dtype)
        self.ks = ks * f * 2 * np.pi
        # define unit cell from generating pattern
        self.unit_cell = Hexagon(f*2/3, init_rot, np.zeros(2))
        # self.inner_hexagon = Hexagon(f / np.sqrt(3), init_rot - 30, np.zeros(2))
        # init trainable phases
        phases = self.unit_cell.sample(ncells)
        self.phases = torch.nn.Parameter(
            torch.tensor(phases, dtype=dtype, requires_grad=True)
        )
        self.relu = torch.nn.ReLU()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.decoder = None

    def forward(self, r, rectify=False):
        """
        Parameters:
            r (nsamples,2): spatial samples
        Returns:
            activity (nsamples,ncells): activity of all cells on spatial samples
        """
        activity = torch.cos((r[:, None] - self.phases[None]) @ self.ks.T)
        activity = torch.sum(activity, axis=-1)  # sum plane waves
        activity = (2 / 3) * (activity / 3 + 0.5)  # Solstad2006 scaling
        activity = self.relu(activity) if rectify else activity
        return activity

    def jacobian(self, r):
        """
        Jacobian of the forward function

        Parameters:
            r (nsamples,2): spatial samples
        Returns:
            J (nsamples,ncells,2): jacobian of the forward function
        """
        relu_grad_mask = self.forward(r) > 0
        J_tmp = -(2 / 9) * torch.sin((r[:, None] - self.phases[None]) @ self.ks.T)
        Jx = torch.sum(J_tmp * self.ks[:, 0], axis=-1)
        Jy = torch.sum(J_tmp * self.ks[:, 1], axis=-1)
        J = torch.stack([Jx, Jy], axis=-1)
        J = relu_grad_mask[..., None] * J
        return J

    def the_jacobian(self, J, sqrt=True):
        """
        Parameters:
            J (nsamples,ncells,2): jacobian
        Returns:
            the jacobian (nsamples,): the jacobian, i.e. sqrt(det(J^T J))
        """
        det = torch.linalg.det(torch.transpose(J, -2, -1) @ J)
        return torch.sqrt(det) if sqrt else det
    
    def decode(self,activity):
        """
        Optimal linear decoding
        """
        if self.decoder is None :
            raise Exception("Must train decoder first, use train_decoder(r)!")
        return activity@self.decoder
    
    def train_decoder(self, r, **kwargs):
        # least squares
        X = self(r,**kwargs)
        self.decoder = torch.linalg.inv((X.T@X))@X.T@r
        return self.decoder

    def loss_fn(self, r):
        """
        Subclass must implement this
        """
        return NotImplemented

    def train_step(self, r):
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss_fn(r)
        loss.backward()
        self.optimizer.step()
        return loss.item()
