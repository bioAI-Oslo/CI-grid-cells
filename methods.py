from typing import Callable
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt


def plot_torei(data, fname, ncols=4, nrows=4, s=1, alpha=0.5):
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, subplot_kw={"projection": "3d"})
    num_plots = ncols * nrows
    azims = np.linspace(0, 360, num_plots // 2 + (num_plots % 2) + 1)[:-1]
    elevs = np.linspace(-90, 90, num_plots // 2 + 1)[:-1]
    view_angles = np.stack(np.meshgrid(azims, elevs), axis=-1).reshape(-1, 2)
    for i, ax in enumerate(axs.flat):
        ax.scatter(xs=data[:, 0], ys=data[:, 1], zs=data[:, 2], s=s, alpha=alpha)
        ax.azim = view_angles[i, 0]
        ax.elev = view_angles[i, 1]
        ax.axis("off")
    fig.savefig(fname)
    return fig, axs


def plot_samples_and_tiling(gridmodule, ratemaps, fname, ratemap_examples=0):
    fig, axs = plt.subplots(ncols=2 + ratemap_examples)
    gridmodule.plot(fig, axs[0])
    axs[0].scatter(*gridmodule.phase_offsets.T, s=5, color="orange", zorder=2)
    axs[0].axis("off")

    for i, ratemap in enumerate(ratemaps[:ratemap_examples]):
        axs[i + 1].imshow(ratemap, origin="lower")
        axs[i + 1].axis("off")

    axs[-1].imshow(np.around(np.sum(ratemaps, axis=0), decimals=10))
    axs[-1].axis("off")
    fig.savefig(fname)
    return fig, axs


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


def grid_cell(
    phase_offset, orientation_offset=0, f=1, non_negative=True, add=True, **kwargs
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
    rot_theta = 60  # degrees
    relative_R = rotation_matrix(rot_theta)
    init_R = rotation_matrix(orientation_offset, **kwargs)

    k1 = np.array([1.0, 0.0])  # init wave vector. unit length in x-direction
    k1 = init_R @ k1
    k2 = relative_R @ k1  # rotate k1 by 60degrees using R
    k3 = relative_R @ k2  # rotate k2 by 60degrees using R
    ks = np.array([k1, k2, k3])
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
            activity = 2 * (activity / 3 + 0.5) / 3
        return activity

    return grid_cell_fn


class GridModule:
    def __init__(
        self, center, orientation_offset=0, f=1, non_negative=True, add=True, **kwargs
    ):
        self.center = center
        self.orientation_offset = orientation_offset
        self.f = f
        self.non_negative = non_negative
        self.add = add

        # define module outer hexagon
        self.outer_radius = 1 / f
        self.outer_hexagon = Hexagon(self.outer_radius, orientation_offset, center)
        # define module inner hexagon based on minimum enclosing circle of Wigner-Seitz cell
        # of a hexagonal lattice with 30 degrees orientation offset to the outer hexagon
        self.inner_radius = 1 / (2 * f * np.cos(np.pi / 6))
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
        self.grid_cell_fn = grid_cell(
            phase_offsets, self.orientation_offset, self.f, self.non_negative, self.add
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
        R_sqrt = R**0.5
        R = R_sqrt * self.inner_radius
        Phi = np.random.uniform(0, 2 * np.pi, N)
        samples[:, 0] = R * np.cos(Phi)
        samples[:, 1] = R * np.sin(Phi)
        return samples

    def plot(self, fig, ax):
        self.inner_hexagon.plot(fig, ax)
        self.outer_hexagon.plot(fig, ax)
        # self.edge_centered_hexagon.plot(fig,ax)
        # self.face_centered_hexagon.plot(fig,ax)
        # ax.scatter(*self.center)


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

    def sample(self, N):
        # sample points within hexagon
        samples = np.zeros((N, 2))
        for i in range(N):
            sample_square = np.random.uniform(-self.radius, self.radius, 2)
            while not self.is_in_hexagon(sample_square):
                sample_square = np.random.uniform(-self.radius, self.radius, 2)
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
