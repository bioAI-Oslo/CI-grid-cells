import numpy as np
import torch
import copy
import tqdm

from utils import *


class Hexagon:
    def __init__(self, radius, orientation_offset, center):
        """
        Parameters:
           radius (float): of the outer (minimal enclosing) circle of the hexagon
           orientation_offset (float): rotate hexagon wrt. cardinal x-direction (degrees)
           center (float): center position of hexagon
        """
        self.radius = radius
        self.orientation_offset = orientation_offset
        self.center = center
        self.area = 3 * np.sqrt(3) * radius * radius / 2
        # create radius and apothem vectors
        self.rotmat_offset = rotation_matrix(orientation_offset, degrees=True)
        self.hpoints = self._init_hexpoints(radius, self.rotmat_offset)
        self.basis = self._init_hexbasis(self.hpoints)

    @staticmethod
    def _init_hexpoints(radius, rotmat_offset):
        """
        Create radius vectors on hexagon

        Parameters:
            radius (float): radius of minimal enclosing hexagon circle
            rotmat_offset (2,2): rotation matrix giving hexagonal rotation offset
        Returns:
            hpoints (6,2): array of hexagona radius vectors, i.e. where the hexagon
                           touches the minimal enclosing circle
        """
        rotmat60 = rotation_matrix(60, degrees=True)
        hpoints = np.array([radius, 0])  # start vector along cardinal x-axis
        hpoints = rotmat_offset @ hpoints
        hpoints = [hpoints]
        for _ in range(5):
            hpoints.append(rotmat60 @ hpoints[-1])
        return np.array(hpoints)

    @staticmethod
    def _init_hexbasis(hpoints):
        """
        Create apothem vectors on hexagon

        Parameters:
            hpoints (6,2): See self._init_hexpoints() for a description
        Returns:
            basis (6,2): Apothem vectors, i.e. the vectors farthest away from
                         the minimal enclosing circle.
        """
        return (np.sqrt(3) * hpoints / 2) @ rotation_matrix(30)

    def is_in_hexagon(self, rs):
        """
        Check if a set of points rs is within hexagon.

        Parameters:
            rs (nsamples,2): points to check if are inside hexagon
        Returns:
            in_hexagon (nsamples,): mask array
        """
        projections = (rs - self.center) @ self.basis.T  # (nsamples,2)
        # all basis vectors have equal length
        in_hexagon = np.max(projections, axis=-1) <= np.sum(self.basis[0] ** 2)
        return in_hexagon

    def sample(self, N, seed=None):
        """
        Vectorized uniform rejection sampling of hexagon using a proposal domain
        define by the minimal enclosing square of the minimal enclosing circle
        of the hexagon.

        Parameters:
            N: (int) number of points to sample
            seed: (int) rng seed
        Returns:
            samples (nsamples,2): array of 2d hexagonal uniform samples
        """
        # sample points within hexagon
        rng = np.random.default_rng(seed)
        missing_samples = N
        samples = np.zeros((N, 2))
        while missing_samples != 0:
            sample_square = rng.uniform(
                -self.radius, self.radius, size=(missing_samples, 2)
            )
            in_hexagon = self.is_in_hexagon(sample_square)
            sample_square = sample_square[in_hexagon]
            samples[
                (N - missing_samples) : (N - missing_samples) + sample_square.shape[0]
            ] = sample_square
            missing_samples -= sample_square.shape[0]
        return samples

    def wrap(self, rs):
        """
        Extends _wrap() to a sequence of points

        Parameters:
            rs (nsamples,2): sequence of 2d points to wrap
        Returns:
            rs_wrapped (nsamples,2):array of wrapped 2d points
        """
        return np.array([self._wrap(rs[i]) for i in range(len(rs))])

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
        if hexagon.is_in_hexagon(x[None])[0]:
            return x - hexagon.center
        hexdrant = np.argmax(hexagon.basis @ (x - hexagon.center))
        hexagon.center += 2 * hexagon.basis[hexdrant]
        return self._wrap(x, hexagon)

    def geodesic(self, p1, p2):
        """
        Parameters:
            p1,p2 (nsamples,2): nsamples of 2D vectors
        """
        p1 = self.wrap(p1)
        p2 = self.wrap(p2)
        p2 = np.concatenate(
            [p2[:, None], p2[:, None] - 2 * self.basis[None]], axis=1
        )  # (nsamples,7,2)
        dist = np.linalg.norm(p2 - p1[:, None], axis=-1)  # (nsamples,7)
        return np.min(dist, axis=-1)

    def mesh(self, n):
        """
        Mesh hexagon. Transforms a scaled meshed rhombus mesh (in rhombus
        coordinates to standard basis) to a hexagonal mesh through wrapping.

        Parameters:
            n: (int) squared mesh resolution (amount of mesh points)
        Returns:
            hexagon_mesh (n**2,2): hexagonal mesh
        """
        # make square mesh based on hexagon size
        square_mesh = np.mgrid[
            self.center[0] : self.center[0] + self.radius * 3 / 2 : complex(n),
            self.center[1] : self.center[1] + self.radius * 3 / 2 : complex(n),
        ].T.reshape(-1, 2)
        # inverse transform square mesh (it is square in rhombus coordinates)
        rhombus_mesh = rhombus_transform(square_mesh)
        # rotate as hexagon is rotated
        rhombus_mesh = rhombus_mesh @ self.rotmat_offset.T
        # wrap rhombus to hexagon
        hexagon_mesh = self.wrap(rhombus_mesh)
        return hexagon_mesh

    def ripleys(
        self, rs, radius, wrap=True, geometric_enclosure="hyperballs", alternative="H"
    ):
        """
        Ripleys k-function counting elements in balls with given radius with
        a periodic hexagonal boundary condition.

        Parameters:
            rs (nsamples,2): spatial/phase positions
            radius (float): ball radius for ripleys k
            wrap (bool): wether to wrap rs to self (hexagon) - this should be true
                         unless they are pre-wrapped.
            geometric_enclosure (str): "hyperballs" or "hexagons" geometric enclosure
            alternative (str): 'K', 'L' or 'H' - defaults to 'H' for corrected
                               and zero-centered expectation. See Kiskowski2009
                               for further explanations.
        Returns:
            ripleys (float): value of statistic
        """
        assert radius <= self.radius, "Larger radius than hexagon enclosing circle"
        rs = self.wrap(rs) if wrap else rs
        # duplicate and tile hexagon with 6 surrounding hexagons, and
        # the corresponding (wrapped) phases.
        outer_rs = rs[None] - 2 * self.basis[:, None]  # (1,n,2) - (6,1,2) => (6,n,2)
        # add inner/surrounded hexagon as index 0
        hexhex_rs = np.concatenate([rs[None], outer_rs], axis=0)  # => (7,n,2)
        # mask of rs that are inside ball with centers given by inner hexagon and radius
        # (7,1,n,2) - (1,n,1,2) => (7,n,n,2)
        diff_rs = hexhex_rs[:, None] - hexhex_rs[:1, :, None]
        if geometric_enclosure == "hyperballs":
            in_geometry = np.linalg.norm(diff_rs, axis=-1) < radius
        elif geometric_enclosure == "hexagons":
            # construct hexagon basis for given ripley radius, (6,2)
            hexbasis = Hexagon._init_hexbasis(self.hpoints * radius / self.radius)
            # vectorised (many hexagons) is_in_hexagon method
            # (7,n,n,2) @ (1,1,2,6) => (7,n,n,6)
            projections = diff_rs @ hexbasis.T[None, None]
            in_geometry = np.max(projections, axis=-1) <= np.sum(
                hexbasis[0] ** 2, axis=-1
            )
        n = rs.shape[0]
        # correct by subtracting n on sum to not count phases defining the ball center
        # this is equivalent to overwriting and setting the diagonal to false.
        ripleys_K = (np.sum(in_geometry) - n) * self.area / (n * (n - 1))
        if alternative == "K":
            return ripleys_K
        ripleys_L = (
            np.sqrt(ripleys_K / np.pi)
            if geometric_enclosure == "hyperballs"
            else np.sqrt(2 * ripleys_K / (3 * np.sqrt(3)))
        )
        if alternative == "L":
            return ripleys_L
        ripleys_H = ripleys_L - radius
        return ripleys_H

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
        self.unit_cell = Hexagon(f * 2 / 3, init_rot, np.zeros(2))
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

    def decode(self, activity):
        """
        Optimal linear decoding
        """
        if self.decoder is None:
            raise Exception("Must train decoder first, use train_decoder(r)!")
        return activity @ self.decoder

    def train_decoder(self, r, **kwargs):
        # least squares
        X = self(r, **kwargs)
        self.decoder = torch.linalg.inv((X.T @ X)) @ X.T @ r
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


def permutation_test(X, Y, statistic, nperms=1000, alternative="two-sided"):
    """
    Permutation test. Method for measuring (under some given statistic function)
    whether two samples X and Y come from the same distribution.

    Parameters:
        X (nsamples,nfeatures): samples of first group
        Y (nsamples,nfeatures): samples of second group
        statistic (Callable): statistic function: (m,n) x (m,n) -> scalar
        nperms (int): number of permutations (samples of null-distribution)
        alternative (str): p-value alternative
    Returns:
        XY_statistic (float): statistic(X,Y)
        pvalue (float): p-value for statistic(X,Y) under null-hypothesis: No difference
                        between the two groups
        H0 (nperms,): array of null-distribution samples
    """
    XY_statistic = statistic(X, Y)
    H0 = np.zeros(nperms)
    N = X.shape[0]
    XY = np.concatenate([X, Y])
    for i in tqdm.trange(nperms):
        XY = np.random.permutation(XY)
        H0[i] = statistic(XY[:N], XY[N:])
    leq = np.sum(XY_statistic <= H0)
    # +1 correction assumes XY_statistic also included in H0
    leq = (leq + 1) / (nperms + 1)
    geq = np.sum(XY_statistic >= H0)
    geq = (geq + 1) / (nperms + 1)
    if alternative == "greater":
        pvalue = geq
    elif alternative == "less":
        pvalue = leq
    else:
        pvalue = min(geq, leq) * 2
    return XY_statistic, pvalue, H0
