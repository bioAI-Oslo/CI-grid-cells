import numpy as np
import scipy
import torch
import copy
import tqdm
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde

from utils import rhombus_transform, rotation_matrix


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

    def surrounding_centers(self, center=None):
        if center is None:
            center = self.center
        scenters = []
        for i in range(6):
            scenters.append(center - 2 * self.basis[i])
        return np.stack(scenters)

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
        old_dist = np.linalg.norm(x - hexagon.center)
        new_center = hexagon.center + 2 * hexagon.basis[hexdrant]
        new_dist = np.linalg.norm(x - new_center)
        if old_dist <= new_dist:
            # numerical imperfection can lead to point landing right outside
            # (between) two hexagons with center +- 2 basis vectors.
            # stop this by choosing the "closest" hexagon
            return x - hexagon.center
        hexagon.center = new_center
        return self._wrap(x, hexagon)

    def geodesic(self, p1, p2):
        """
        Shortest distance between two points on a 2D sheet with hexagonal
        periodic boundary conditions.

        Parameters:
            p1,p2 (nsamples,2): nsamples of 2D vectors
        Returns:
            geodesic (nsamples,): shortest distance between p1 and p2
        """
        p1 = self.wrap(p1)
        p2 = self.wrap(p2)
        p2 = np.concatenate(
            [p2[:, None], p2[:, None] - 2 * self.basis[None]], axis=1
        )  # (nsamples,7,2)
        dist = np.linalg.norm(p2 - p1[:, None], axis=-1)  # (nsamples,7)
        return np.min(dist, axis=-1)

    def mesh(self, n, wrap=True):
        """
        Mesh hexagon by inverting a mesh in rhombus coordinates to the
        standard basis. Wrap mesh to the unit cell.

        Parameters:
            n: (int) squared mesh resolution (amount of mesh points)
        Returns:
            hexagon_mesh (n**2,2): hexagonal mesh
        """
        # make square mesh based on hexagon size
        square_mesh = np.mgrid[
            self.center[0] : self.center[0] + self.radius * 3 / 2 : complex(n + 1),
            self.center[1] : self.center[1] + self.radius * 3 / 2 : complex(n + 1),
        ]  # .T.reshape(-1, 2)
        square_mesh = square_mesh[:, :-1, :-1].reshape(2, -1).T
        # inverse transform square mesh (it is square in rhombus coordinates)
        rhombus_mesh = rhombus_transform(square_mesh)
        # rotate as hexagon is rotated
        rhombus_mesh = rhombus_mesh @ self.rotmat_offset.T
        if wrap:
            # wrap rhombus to hexagon
            hexagon_mesh = self.wrap(rhombus_mesh)
            return hexagon_mesh
        return rhombus_mesh

    def plot(self, fig=None, ax=None, center=None, colors=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        center = self.center if center is None else center
        hpoints = self.hpoints + center
        for i in range(len(hpoints)):
            line_segment = np.stack([hpoints[i], hpoints[(i + 1) % 6]])
            if not (colors is None):
                ax.plot(*line_segment.T, color=colors[i], **kwargs)
            else:
                ax.plot(*line_segment.T, **kwargs)
        # ax.set_aspect("equal")
        return fig, ax


class HexagonalGCs(torch.nn.Module):
    """
    torch model for learning optimal grid cell phases
    """

    def __init__(
        self,
        ncells=3,
        f=1,
        init_rot=0,
        shift=0,
        dropout=False,
        lr=1e-3,
        dtype=torch.float32,
        seed=None,
        **kwargs
    ):
        super(HexagonalGCs, self).__init__(**kwargs)
        # init static grid properties
        self.ncells, self.f, self.init_rot, self.dtype = ncells, f, init_rot, dtype
        self.lr, self.shift = lr, shift
        rotmat_init = rotation_matrix(init_rot)
        rotmat_60 = rotation_matrix(60)
        k1 = np.array([1.0, 0.0])
        k1 = rotmat_init @ k1
        ks = np.array([np.linalg.matrix_power(rotmat_60, k) @ k1 for k in range(3)])
        ks = torch.tensor(ks, dtype=dtype)
        self.ks = ks * f * 2 * np.pi
        # define unit cell from generating pattern
        self.unit_cell = Hexagon(2 / (3 * f), init_rot, np.zeros(2))
        # init trainable phases
        phases = self.unit_cell.sample(
            ncells, seed
        )  # default random uniform initial phases
        self.set_phases(phases)  # initialises trainable params and optimizer
        self.relu = torch.nn.ReLU()
        self.decoder = None
        self.dropout = torch.nn.Dropout() if dropout else None

    def set_phases(self, phases):
        """
        Initialises trainable phases and optimizer

        Parameters:
            phases (ncells,2): Sequence (np.array, torch tensor, list etc.) of
                               initial/overwritten phases
        """
        phases = (
            torch.tensor(phases, dtype=self.dtype)
            if not isinstance(phases, torch.Tensor)
            else phases
        )
        self.phases = torch.nn.Parameter(phases.clone(), requires_grad=True)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.ncells = len(phases)
        return None

    @staticmethod
    def jitter(nsamples, magnitudes=None, magnitude=1e-2, epsilon=1e-8):
        """
        Parameters:
            nsamples int: size of jitter vector
            magnitudes (nsamples,): ndarray of optional jitter magnitudes
            magnitude float: magnitude range to sample

        Return:
            v_jitter (nsamples,2): The jitter vectors
        """
        thetas = torch.rand(size=(nsamples,)) * 2 * np.pi
        if magnitudes is None:
            magnitudes = torch.rand(size=(nsamples,)) * magnitude + epsilon
        v_jitter = torch.stack([torch.cos(thetas), torch.sin(thetas)], axis=-1)
        v_jitter = magnitudes[:, None] * v_jitter
        return v_jitter, magnitudes

    def forward(self, r, dp=None):
        """
        Parameters:
            r (nsamples,2): spatial samples
            dp (ncells,2): optional phase (jitter) shift
        Returns:
            activity (nsamples,ncells): activity of all cells on spatial samples
        """
        phases = self.phases + dp if dp is not None else self.phases
        activity = torch.cos((r[:, None] - phases[None]) @ self.ks.T)
        activity = torch.sum(activity, axis=-1)  # sum plane waves
        # activity = (2 / 3) * (activity / 3 + 0.5)  # Solstad2006 rescaling, range: [-1.5,3] -> [0,1]
        activity = (
            2 * activity / 9 + 1 / 3
        )  # Solstad2006 rescaling, range: [-1.5,3] -> [0,1]
        activity = activity - self.shift  # shift to range: [-shift,1-shift]
        # activity = activity + dp if dp is not None else activity
        activity = (
            self.relu(activity) if self.shift else activity
        )  # rectify, range: [0,1-shift]
        return activity

    def jacobian(self, r, dp=None):
        """
        Jacobian of the forward function

        Parameters:
            r (nsamples,2): spatial samples
            dp (ncells,2): optional phase (jitter) shift
        Returns:
            J (nsamples,ncells,2): jacobian of the forward function
        """
        phases = self.phases + dp if dp is not None else self.phases
        J_tmp = -(2 / 9) * torch.sin((r[:, None] - phases[None]) @ self.ks.T)
        Jx = torch.sum(J_tmp * self.ks[:, 0], axis=-1)
        Jy = torch.sum(J_tmp * self.ks[:, 1], axis=-1)
        if self.dropout:
            mask = self.dropout(torch.ones_like(Jx))
            Jx, Jy = Jx * mask, Jy * mask
        J = torch.stack([Jx, Jy], axis=-1)
        if self.shift:
            relu_grad_mask = self.forward(r, dp) > 0
            J = relu_grad_mask[..., None] * J
        return J

    def the_jacobian(self, J, sqrt=True):
        """
        Parameters:
            J (nsamples,ncells,2): jacobian
        Returns:
            the jacobian (nsamples,): the jacobian, i.e. sqrt(det(J^T J))
        """
        det = torch.linalg.det(self.metric_tensor(J))
        return torch.sqrt(det) if sqrt else det

    def metric_tensor(self, J):
        """
        Parameters:
            J (nsamples,ncells,2): jacobian
        Returns:
           metric tensor (nsamples,2,2): the metric tensor
        """
        return torch.transpose(J, -2, -1) @ J

    def CI_metric(self, Gs=None, J=None, r=None):
        """
        Conformal Isometry metric

        Parameters:
            Gs (nsamples,2,2): metric tensor
            J (nsamples,ncells,2): jacobian
            r (nsamples,2): spatial samples
        Returns:
            CI_metric float: the conformal isometry metric
        """
        if Gs is None:
            if J is None:
                r = (
                    torch.tensor(r, dtype=self.dtype)
                    if not isinstance(r, torch.Tensor)
                    else r
                )
                J = self.jacobian(r)
            Gs = self.metric_tensor(J)
        g11 = Gs[:, 0, 0]
        g22 = Gs[:, 1, 1]
        g12 = Gs[:, 0, 1]
        return (
            torch.var(g11)
            + torch.var(g22)
            + torch.mean((g11 - g22) ** 2)
            + 2 * torch.mean(g12**2)
        )

    def decode(self, activity):
        """
        Optimal linear decoding

        Parameters:
            activity (nsamples,ncells): grid cell activity across spatial samples
        Returns:
            r_pred (nsamples,2): predicted spatial positions
        """
        if self.decoder is None:
            raise Exception("Must train decoder first, use train_decoder(r)!")
        if self.decoder.shape[0] > activity.shape[1]:
            # make activity (nsamples,ncells+1)
            activity = torch.concatenate(
                [torch.ones((activity.shape[0], 1)), activity], axis=-1
            )  # add bias
        return activity @ self.decoder

    def train_decoder(self, r, bias=True, **kwargs):
        """
        Least squares (linear) decoder

        Parameters:
            r (nsamples,2): spatial positions
        Returns:
            decoder (ncells(+1),2): Linear decoder including optinal bias
        """
        # least squares
        X = self(r, **kwargs)  # (nsamples,ncells)
        if bias:
            X = torch.concatenate([torch.ones((X.shape[0], 1)), X], axis=-1)  # add bias
        self.decoder = torch.linalg.lstsq(X, r).solution
        # self.decoder = torch.linalg.inv((X.T @ X)) @ X.T @ r
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

    def phase_kde(
        self,
        phases=None,
        res=64,
        unit_cell=None,
        double_extension=True,
        triple_extension=False,
        wrap=True,
        **kwargs
    ):
        """Approximate KDE of phases by retiling of unit cell
        Args:
            phases: array of phases, of shape (N, 2) where N is number of units.
            res (int): resolution of grid
            unit_cell: unit cell class
            double_extension (bool): whether to extend the unit cell with one (False) or two (True) cells in each direction
            wrap (bool): whether to wrap phases to unit cell (should always be done, unless done beforehand)
        Returns:
            Estimated kernel, see scipy.stats.gaussian_kde for usage
        """
        phases = self.phases.detach().numpy() if phases is None else phases
        phases = self.unit_cell.wrap(phases) if wrap else phases
        unit_cell = self.unit_cell if unit_cell is None else unit_cell
        phase_tiles = [phases - 2 * unit_cell.basis[i] for i in range(6)]
        if double_extension:
            phase_tiles += [phases - 3 * unit_cell.hpoints[i] for i in range(6)]
            phase_tiles += [phases - 4 * unit_cell.basis[i] for i in range(6)]
        if triple_extension:
            phase_tiles += [phases - 6 * unit_cell.basis[i] for i in range(6)]
            phase_tiles += [phases - 6 * unit_cell.basis[i] - 2*unit_cell.basis[(i+2)%6] for i in range(6)]
            phase_tiles += [phases - 6 * unit_cell.basis[i] - 4*unit_cell.basis[(i+2)%6] for i in range(6)]
        expanded_phases = np.concatenate((phases, *phase_tiles), axis=0)
        kernel = gaussian_kde(expanded_phases.T, **kwargs)
        kde, mesh = None, None
        if res is not None:
            # use kde on mesh
            mesh = unit_cell.mesh(res)
            kde = kernel(mesh.T)
        return kde, mesh, kernel, expanded_phases

    def phase_kde_autocorrelogram(self, phases=None, res=64, bw_method=0.1):
        """
        Compute autocorrelogram of KDE of a set of phases.

        Parameters:
            phases: array of phases, of shape (N, 2).
            res (int): resolution of grid
            bw_method (float): bandwidth for KDE
        Returns:
            autocorr (res,res): autocorrelogram
            ratemap (res,res): KDE evaluated on a square grid
            square_mesh (res,res,2): square grid
            large_ratemap (large_res,large_res): KDE evaluated on a larger grid
            large_square_mesh (large_res,large_res,2): larger square grid
        """
        phases = self.phases.detach().clone().numpy() if phases is None else phases
        # compute KDE
        kernel = self.phase_kde(phases, res=None, double_extension=True, bw_method=bw_method)[2]
        # create square grid and mask based on unit cell
        bins = np.linspace(-self.unit_cell.radius, self.unit_cell.radius, res)
        xx, yy = np.meshgrid(bins, bins)
        square_mesh = np.stack((xx, yy), axis=-1)  # (res,res,2)
        # evaluate kernel
        ratemap = kernel(square_mesh.reshape(-1, 2).T).reshape(res, res)
        # evaluate kernel on a larger grid to compute 'valid' autocorrelation
        # with the same shape as ratemap
        large_res = 2 * res - 1
        bins = np.linspace(
            -self.unit_cell.radius * 3 / 2, self.unit_cell.radius * 3 / 2, large_res
        )
        xx, yy = np.meshgrid(bins, bins)
        large_square_mesh = np.stack((xx, yy), axis=-1)  # (res,res,2)
        large_ratemap = kernel(large_square_mesh.reshape(-1, 2).T).reshape(
            large_res, large_res
        )
        # autocorrelate
        autocorr = corrcoef_valid2d(
            ratemap, large_ratemap
        )  # Pearson (spatial) correlation
        return autocorr, ratemap, square_mesh, large_ratemap, large_square_mesh

    def grid_score(self, phases=None, slice_inner_circle=True, res=64, bw_method=0.1):
        """Compute grid score of the KDE of a set of phases.
        Parameters:
            phases: array of phases, of shape (N, 2) where N is number of units.
            rotation: rotation angle in degrees to rotate phases initially
            res (int): resolution of grid
            bw_method (float): bandwidth for KDE
        Returns:
            gcs (float): grid cell score
        """
        autocorr, _, square_mesh, _, _ = self.phase_kde_autocorrelogram(phases, res, bw_method=bw_method)
        # outer circle mask
        mask = np.linalg.norm(square_mesh, axis=-1) < self.unit_cell.radius
        # slice out inner circle
        if slice_inner_circle:
            inner_circle_mask = np.ones_like(mask)
            inner_circle_mask[np.linalg.norm(square_mesh, axis=-1) < bw_method] = 0
            mask = np.logical_and(mask, inner_circle_mask)
        # compute correlations
        correlates = []
        angles = range(30, 180 + 30, 30)
        for angle in angles:
            rotated_autocorr = scipy.ndimage.rotate(
                autocorr, angle, reshape=False, order=0
            )
            correlate = np.corrcoef(rotated_autocorr[mask], autocorr[mask])[0, 1]
            correlates.append(correlate)
        # extract 30 and 60 degree correlations
        r30 = correlates[::2]
        r60 = correlates[1::2]
        gcs = np.mean(r60) - np.mean(r30)  # range: [-2,2]
        return gcs

    def ripleys(self, radius, phases=None):
        """
        Ripleys-H function counting elements in balls with given radius with
        a periodic hexagonal boundary condition. See Kiskowski2009 for further
        explanations.

        Parameters:
            radius (float): ball radius for ripleys k
            phases (nsamples,2): phase positions
        Returns:
            ripleys-H (float): value of statistic
        """
        assert (
            radius <= self.unit_cell.radius
        ), "Larger radius than hexagon enclosing circle"
        phases = self.phases.detach().clone().numpy() if phases is None else phases
        phases = self.unit_cell.wrap(phases)
        # duplicate and tile hexagon with 6 surrounding hexagons, and
        # the corresponding (wrapped) phases.
        outer_phases = (
            phases[None] - 2 * self.unit_cell.basis[:, None]
        )  # (1,n,2) - (6,1,2) => (6,n,2)
        # add inner/surrounded hexagon as index 0
        hexhex_phases = np.concatenate(
            [phases[None], outer_phases], axis=0
        )  # => (7,n,2)
        # mask of rs that are inside ball with centers given by inner hexagon and radius
        # (7,1,n,2) - (1,n,1,2) => (7,n,n,2)
        diff_phases = hexhex_phases[:, None] - hexhex_phases[:1, :, None]
        in_geometry = np.linalg.norm(diff_phases, axis=-1) < radius
        n = phases.shape[0]
        # correct by subtracting n on sum to not count phases defining the ball center
        # this is equivalent to overwriting and setting the diagonal to false.
        ripleys_K = (np.sum(in_geometry) - n) * self.unit_cell.area / (n * (n - 1))
        ripleys_L = np.sqrt(ripleys_K / np.pi)
        ripleys_H = ripleys_L - radius
        return ripleys_H


def corrcoef_valid2d(small_map, large_map):
    """
    Compute Pearson correlation coefficient between small_map and all
    possible slices of large_map with the same size as small_map.

    Parameters:
        small_map (I,J): small map
        large_map (I',J'): large map
    Returns:
        out_map (I' - I + 1, J' - J + 1): map of correlations
    """
    Is, Js = small_map.shape
    Il, Jl = large_map.shape
    out_map = np.zeros((Il - Is + 1, Jl - Js + 1))
    for i in range(Il - Is + 1):
        for j in range(Jl - Js + 1):
            # Pearson correlation
            out_map[i, j] = np.corrcoef(
                small_map.flatten(), large_map[i : i + Is, j : j + Js].flatten()
            )[0, 1]
            # Cosine similarity
            #out_map[i, j] = np.dot(small_map.flatten(), large_map[i : i + Is, j : j + Js].flatten()) / (
            #    np.linalg.norm(small_map.flatten()) * np.linalg.norm(large_map[i : i + Is, j : j + Js].flatten())
            #)
    return out_map


def permutation_test(
    rvs_X, rvs_Y, statistic=None, nperms=1000, alternative="two-sided"
):
    """
    Permutation test. Method to determine whether two samples
    X and Y come from the same distribution.

    Parameters:
        rvs_X (nsamples1,): response values of first group
        rvs_Y (nsamples2,): response values of second group
        statistic (Callable): statistic function: (nsamples1,) x (nsamples2,) -> scalar
        nperms (int): number of permutations (samples of null-distribution)
        alternative (str): p-value alternative
    Returns:
        XY_statistic (float): statistic(X,Y)
        pvalue (float): p-value for statistic(X,Y) under null-hypothesis: No difference
                        between the two groups
        H0 (nperms,): array of null-distribution samples
    """
    if statistic is None:
        statistic = lambda X, Y: np.mean(X) - np.mean(Y)
    XY_statistic = statistic(rvs_X, rvs_Y)
    H0 = np.zeros(nperms)
    XY = np.concatenate([rvs_X, rvs_Y])
    for i in tqdm.trange(nperms):
        XY = np.random.permutation(XY)
        H0[i] = statistic(XY[: len(rvs_X)], XY[len(rvs_X) :])
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
        pvalue = 2.0 * min(leq, 1 - leq)
    return XY_statistic, pvalue, H0


def phase_kde(unit_cell, phases, **kwargs):
    """Approximate KDE of phases by retiling of unit cell
    Args:
        unit_cell: unit cell class
        phases: array of phases, of shape (N, 2) where N is number of units.
    Returns:
        Estimated kernel, see scipy.stats.gaussian_kde for usage
    """
    phase_tiles = [phases - 2 * unit_cell.basis[i] for i in range(6)]
    expanded_phases = np.concatenate((phases, *phase_tiles), axis=0)
    kernel = gaussian_kde(expanded_phases.T, **kwargs)
    return kernel, expanded_phases


def activity_kde(model):
    def kernel(r):
        activity = model(torch.tensor(r, dtype=torch.float32)).detach().numpy()
        return np.sum(activity, axis=-1)

    return kernel
