from typing import Callable
import numpy as np
import numpy.ma as ma
import numpy.linalg as npl
import matplotlib.pyplot as plt


def rotation_matrix(theta, degrees=True, **kwargs) -> np.ndarray:
    """
    Creates a 2D rotation matrix for theta
    Parameters
    ----------
    theta : float
        angle offset wrt. the cardinal x-axis
    degrees : boolean
        Whether to use degrees or radians
    Returns
    -------
    rotmat : np.ndarray
        the 2x2 rotation matrix
    Examples
    --------
    >>> import numpy as np
    >>> x = np.ones(2) / np.sqrt(2)
    >>> rotmat = rotation_matrix(45)
    >>> tmp = rotmat @ x
    >>> eps = 1e-8
    >>> np.sum(np.abs(tmp - np.array([0., 1.]))) < eps
    True
    """
    # convert to radians
    theta = theta * np.pi / 180 if degrees else theta
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))


def torus(R=1, r=0.5, alpha=0, beta=0, n=100, a=1, b=1):
    """
    Generate a torus with radius R and inner radius r
    R > r => ring-torus (standard), R = r => Horn-torus, R < r => Spindel-torus

    params:
        R: radius of outer circle
        r: radius of inner circle
        n: square root of number of points on torus
        alpha: outer circle twist parameter wrt. inner circle. (twisted torus)
        beta: inner circle twist parameter wrt. outer circle. (unknown)
    """
    theta = np.linspace(-a * np.pi, a * np.pi, n)  # +np.pi/2
    phi = np.linspace(-b * np.pi, b * np.pi, n)
    x = (R + r * np.cos(theta[None] - alpha * phi[:, None])) * np.cos(
        phi[:, None] - beta * theta[None]
    )
    y = (R + r * np.cos(theta[None] - alpha * phi[:, None])) * np.sin(
        phi[:, None] - beta * theta[None]
    )
    z = r * np.sin(theta[None] - alpha * phi[:, None])
    coords = np.array([x, y, z]).T
    return coords


def grid_cell(
    phase_offset,
    orientation_offset=0,
    f=1,
    rot_theta=60,
    n_comps=3,
    non_negative=True,
    add=True,
    **kwargs
) -> Callable:
    """
    Grid cell pattern constructed from three interacting 2D (plane) vectors
    with 60 degrees relative orientational offsets.
    See e.g. the paper: "From grid cells to place cells: A mathematical model"
    - Moser & Einevoll 2005
    Parameters
    ----------
    phase_offset : np.ndarray
        2D-array. Spatial (vector) phase offset of the grid pattern. Note that
        the grid pattern is np.array([f,f])-periodic, so phase-offsets are also
        f-periodic
    orientation_offset : float
        First plane vector is default along the cardinal x-axis. Phase-offset
        turns this plane vector counter clockwise (default in degrees, but
        can use **kwargs - degrees=False to use radians)
    f : float
        Spatial frequency / periodicity. f=1 makes the grid cell unit-periodic
    Returns
    -------
    grid_cell_fn : function
        A grid cell function which can be evaluated at locations r
    Examples
    --------
    >>> import numpy as np
    >>> x = np.zeros(2)
    >>> gc = grid_cell()
    >>> gc(x)
    3.0
    """
    relative_R = rotation_matrix(rot_theta)
    init_R = rotation_matrix(orientation_offset, **kwargs)

    k1 = np.array([1.0, 0.0])  # init wave vector. unit length in x-direction
    k1 = init_R @ k1
    ks = np.array([npl.matrix_power(relative_R, k) @ k1 for k in range(n_comps)])
    ks *= 2 * np.pi  # spatial angular frequency (unit-movement in space is one period)
    ks *= f  # user-defined spatial frequency

    def grid_cell_fn(r):
        """
        Grid cell function with fixed parameters given by outer function. I.e.
        a mapping from some spatial coordinates r to grid cell activity.
        Parameters
        ----------
        r : np.ndarray
            [1,2 or 3]D array. For a 3D array, the shape is typically (Ng,Ng,2).
            A tensor of 2D-spatial coordinates.
        Returns:
        grid_cell_activity: np.ndarray or float
            [0,1 or 2]D array. For a 2D array, the shape is typically (Ng,Ng).
            The activity of this grid cell across all spatial coordinates in
            the grid (Ng,Ng).
        """
        r0 = phase_offset
        if r0.ndim == 2 and r.ndim > 2:
            for i in range(1, r.ndim):
                r0 = r0[:, None]
        if not add:
            return np.cos((r - r0) @ ks.T)

        activity = np.sum(np.cos((r - r0) @ ks.T), axis=-1)
        if non_negative:
            activity = np.maximum(activity, 0)
        else:
            # scale to [0,1]
            activity = (activity / (2*n_comps) + 0.5)
        return activity

    return grid_cell_fn


"""
    For hexagonal grid cells we can use the lattice constant of the 
    hexagonal lattice that is formed by the ratemaps maxima as a more
    intuitive way of parameterizing the pattern for generation.
"""

def hex_grid_cell(
    phase_offset,
    orientation_offset=0,
    a=1, # lattice constant
    non_negative=True,
    add=True,
    **kwargs
) -> Callable:
    """
    Grid cell pattern constructed from three interacting 2D (plane) vectors
    with 60 degrees relative orientational offsets.
    See e.g. the paper: "From grid cells to place cells: A mathematical model"
    - Moser & Einevoll 2005
    Parameters
    ----------
    phase_offset : np.ndarray
        2D-array. Spatial (vector) phase offset of the grid pattern. Note that
        the grid pattern is np.array([f,f])-periodic, so phase-offsets are also
        f-periodic
    orientation_offset : float
        First plane vector is default along the cardinal x-axis. Phase-offset
        turns this plane vector counter clockwise (default in degrees, but
        can use **kwargs - degrees=False to use radians)
    f : float
        Spatial frequency / periodicity. f=1 makes the grid cell unit-periodic
    Returns
    -------
    grid_cell_fn : function
        A grid cell function which can be evaluated at locations r
    Examples
    --------
    >>> import numpy as np
    >>> x = np.zeros(2)
    >>> gc = grid_cell()
    >>> gc(x)
    3.0
    """

    # the ratemap maxima pattern is rotated against the k1 by about 30 deg
    # to correct for zero orientation w.r.t the actual pattern instead of
    # the generators, we need to correct the orientation_offset by these 30 deg
    relative_R = rotation_matrix(60., degrees=True)
    init_R = rotation_matrix(orientation_offset - 30., degrees=True, **kwargs)

    k1 = np.array([1.0, 0.0])  # init wave vector. unit length in x-direction
    k1 = init_R @ k1
    ks = np.array([npl.matrix_power(relative_R, k) @ k1 for k in range(3)])
    ks *= 2 * np.pi  # spatial angular frequency (unit-movement in space is one period)

    # translate user-defined lattice constant a into spatial frequency for 
    # the generating plane waves
    # a_GC = 1 / (f*cos(30.)) = 2 / (f*sqrt(3))
    # -> f = 2 / (a*sqrt(3))
    ks *= 2 / (a*np.sqrt(3)) # user-defined spatial frequency

    def grid_cell_fn(r):
        """
        Grid cell function with fixed parameters given by outer function. I.e.
        a mapping from some spatial coordinates r to grid cell activity.
        Parameters
        ----------
        r : np.ndarray
            [1,2 or 3]D array. For a 3D array, the shape is typically (Ng,Ng,2).
            A tensor of 2D-spatial coordinates.
        Returns:
        grid_cell_activity: np.ndarray or float
            [0,1 or 2]D array. For a 2D array, the shape is typically (Ng,Ng).
            The activity of this grid cell across all spatial coordinates in
            the grid (Ng,Ng).
        """
        r0 = phase_offset
        if r0.ndim == 2 and r.ndim > 2:
            for i in range(1, r.ndim):
                r0 = r0[:, None]
        if not add:
            return np.cos((r - r0) @ ks.T)

        activity = np.sum(np.cos((r - r0) @ ks.T), axis=-1)
        if non_negative:
            activity = np.maximum(activity, 0)
        else:
            # scale to [0,1]
            activity = 2 * (activity / 3 + 0.5) / 3
        return activity

    return grid_cell_fn


class GridModule:
    def __init__(
        self, center, orientation_offset=0, a=1, non_negative=True, add=True, **kwargs
    ):
        self.center = center
        self.orientation_offset = orientation_offset
        self.a = a
        self.non_negative = non_negative
        self.add = add

        # define module outer hexagon
        self.outer_radius = self.a
        self.outer_hexagon = Hexagon(self.outer_radius, orientation_offset, center)
        # define module inner hexagon based on minimum enclosing circle of Wigner-Seitz cell
        # of a hexagonal lattice with 30 degrees orientation offset to the outer hexagon
        self.inner_radius = self.a / np.sqrt(3)
        self.inner_hexagon = Hexagon(self.inner_radius, orientation_offset + 30, center)

        # define phase offsets of minimal grid module for tiling space?
        self.edge_centered_hexagon = Hexagon(
            self.inner_radius / 2, orientation_offset + 30, center
        )
        self.face_centered_hexagon = Hexagon(
            self.outer_radius / 4, orientation_offset, center
        )

    def init_module(self, phase_offsets):
        self.phase_offsets = phase_offsets
        self.grid_cell_fn = hex_grid_cell(
            phase_offset=phase_offsets,
            orientation_offset=self.orientation_offset,
            a=self.a,
            non_negative=self.non_negative,
            add=self.add,
        )

    def __call__(self, r):
        return self.grid_cell_fn(r)

    def sample_minimal_phase_offsets(self, n_samples):
        n_samples = n_samples - n_samples % 3  # make sure n_samples is divisible by 3
        total_samples = np.zeros((n_samples, 2))
        center_samples = self.inner_hexagon.sample(n_samples // 3)
        for i, center_sample in enumerate(center_samples):
            total_samples[i * 3 : (i + 1) * 3] = (
                self.inner_hexagon.hpoints[::2] * 2 / 3 + center_sample
            )
        return total_samples

    def sample_four_phase_offsets(self, n_samples):
        total_samples = np.zeros((n_samples, 2))
        center_samples = self.inner_hexagon.sample(n_samples // 4)
        total_samples[::4] = center_samples
        for i, center_sample in enumerate(center_samples):
            for j, hpoint in enumerate(self.inner_hexagon.hpoints[::2]):
                total_samples[i * 4 + j + 1] = hpoint + center_sample
        return total_samples

    def sample_phase_offsets_disk(self, N):
        # sample points within hexagon
        samples = np.zeros((N, 2))
        R = np.random.uniform(0, 1, N)
        R_sqrt = R ** 0.5
        R = R_sqrt * self.inner_radius
        Phi = np.random.uniform(0, 2 * np.pi, N)
        samples[:, 0] = R * np.cos(Phi)
        samples[:, 1] = R * np.sin(Phi)
        return samples

    def plot(self, fig=None, ax=None, **kwargs):
        if fig == None or ax == None:
            fig,ax = plt.subplots(**kwargs)
        self.inner_hexagon.plot(fig, ax)
        self.outer_hexagon.plot(fig, ax)
        ax.scatter(*self.phase_offsets.T)
        # self.edge_centered_hexagon.plot(fig,ax)
        # self.face_centered_hexagon.plot(fig,ax)
        # ax.scatter(*self.center)
        return fig,ax

    def period_mask(self, board, center_board_pixels=False):
        """
        Return a mask of the board where the grid module is periodic.
        Parameters
        ----------
        board : np.ndarray
            [Nx,Ny,2] array of floats.
        Returns
        -------
        mask : np.ndarray
            [Nx,Ny] array of bools.
        """
        mask = np.zeros(board.shape[:2], dtype=bool)
        pixel_center_x = (board[0, 1, 0] - board[0, 0, 0]) / 2
        pixel_center_y = (board[1, 0, 1] - board[0, 0, 1]) / 2
        pixel_center = np.array([pixel_center_x, pixel_center_y])
        board = board + pixel_center if center_board_pixels else board
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                mask[i, j] = self.inner_hexagon.is_in_hexagon(board[i, j])
        return mask

    def period_mask_DEPRECATED(self, board):
        """
        Return a mask of the board where the grid module is periodic.

        OBS! This is a hexagon with hpoints half the distance to the outer hexagon.
        This is not the periodic (unit) cell. The periodic cell is the wigner setiz cell

        Parameters
        ----------
        board : np.ndarray
            [Nx,Ny,2] array of floats.
        Returns
        -------
        mask : np.ndarray
            [Nx,Ny] array of bools.
        """
        mask = np.zeros(board.shape[:2], dtype=bool)
        space_hexagon = Hexagon(1 / (2 * self.f), self.orientation_offset, self.center)
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                mask[i, j] = space_hexagon.is_in_hexagon(board[i, j])
        return mask

    def masked_ratemaps(self, board):
        ratemaps = self(board)
        mask = self.period_mask(board)
        return ma.masked_array(
            ratemaps, mask=np.repeat(~mask[None], ratemaps.shape[0], axis=0)
        )


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
        #attempt to use inner hexagonal circle instead
        #hpoints = (rotmat60 @ hpoints[0] - hpoints[0]) / 2 + hpoints[0]
        #hpoints = [hpoints]
        for _ in range(5):
            hpoints.append(rotmat60 @ hpoints[-1])
        self.hpoints = np.array(hpoints)

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

    def plot(self, fig, ax, color="blue"):
        hpoints = self.hpoints + self.center
        for i in range(len(hpoints)):
            ax.plot(*hpoints[i : (i + 2)].T, color=color)
        last_line = np.array([hpoints[-1], hpoints[0]])
        ax.plot(*last_line.T, color=color)

        ax.set_aspect("equal")
        return fig, ax


class SquareGridModule:
    def __init__(self, center, orientation_offset, f, non_negative, add):
        self.center = center
        self.f = f
        self.orientation_offset = orientation_offset
        self.non_negative = non_negative
        self.add = add

        # define module outer hexagon
        self.outer_radius = 1 / f
        self.inner_radius = self.outer_radius / 2

        self.inner_square = Square(
            self.inner_radius, self.orientation_offset, self.center
        )
        self.outer_square = Square(
            self.outer_radius, self.orientation_offset, self.center
        )

        self.phase_offsets = None

    def init_module(self, phase_offsets):
        self.phase_offsets = phase_offsets
        self.grid_cell_fn = grid_cell(
            phase_offset=phase_offsets,
            orientation_offset=self.orientation_offset,
            f=self.f,
            rot_theta=90,
            n_comps=2,
            non_negative=self.non_negative,
            add=self.add,
        )

    def __call__(self, r):
        return self.grid_cell_fn(r)

    def plot(self, fig=None, ax=None, **kwargs):
        if fig is None or ax is None:
            fig,ax = plt.subplots(**kwargs)
        self.inner_square.plot(fig, ax)
        self.outer_square.plot(fig, ax)
        if self.phase_offsets is not None:
           ax.scatter(*self.phase_offsets.T)
        return fig, ax


class Square:
    def __init__(self, radius, orientation_offset, center):
        self.radius = radius
        self.orientation_offset = orientation_offset
        self.center = center
        self.area = radius * radius

        # create hexagonal points
        rotmat90 = rotation_matrix(90, degrees=True)
        rotmat_offset = rotation_matrix(orientation_offset, degrees=True)
        hpoints = np.array([radius, radius])  # start vector along diagonal
        hpoints = rotmat_offset @ hpoints
        hpoints = [hpoints]
        for _ in range(3):
            hpoints.append(rotmat90 @ hpoints[-1])
        self.hpoints = np.array(hpoints)

    def sample(self, N, seed=None):
        # sample points within hexagon
        rng = np.random.default_rng(seed)
        return rng.uniform(-self.radius, self.radius, (N, 2))

    def plot(self, fig, ax, color="blue"):
        hpoints = self.hpoints + self.center
        for i in range(len(hpoints)):
            ax.plot(*hpoints[i : (i + 2)].T, color=color)
        last_line = np.array([hpoints[-1], hpoints[0]])
        ax.plot(*last_line.T, color=color)

        ax.set_aspect("equal")
        return fig, ax


def intersect(
    u1, v1, u2, v2, constraint1=[-np.inf, np.inf], constraint2=[-np.inf, np.inf]
):
    """
    Calculate intersection of two line segments defined as:
    l1 = {u1 + t1*v1 : u1,v1 in R^n, t1 in constraint1 subseq R},
    l2 = {u2 + t2*v2 : u2,v2 in R^n, t2 in constraint2 subseq R}
    Args:
        u1: bias of first line-segment
        v1: "slope" of first line-segment
        u2: bias of second line-segment
        v2: "slope" of first line-segment
        constraint1: 2d-array(like) of boundary points
                     for the "t-values" of the first line-segment
        constraint1: 2d-array(like) of boundary points
                     for the "t-values" of the second line-segment
    """
    matrix = np.array([v1, -v2]).T
    vector = u2 - u1
    try:
        solution = np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError as e:
        # Singular matrix (parallell line segments)
        print(e)
        return None, False

    # check if solution satisfies constraints
    if (constraint1[0] <= solution[0] <= constraint1[1]) and (
        constraint2[0] <= solution[1] <= constraint2[1]
    ):
        return u1 + solution[0] * v1, True

    return u1 + solution[0] * v1, False


def get_intervals(pdiag, feat_list):
    top_features = {}
    for i in range(len(feat_list)):
        temp_feat = np.sort(pdiag[i][:, 1] - pdiag[i][:, 0])
        top_features[str(i)] = []
        for j in range(feat_list[i]):
            try:
                top_features[str(i)].append(temp_feat[-1 - j])
            except:
                top_features[str(i)].append(0)
    return top_features
