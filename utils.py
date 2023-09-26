import numpy as np
import torch


def create_board(nx=1, ny=None, res=64, to_torch_tensor=True):
    """
    Create a meshgrid
    """
    if ny is None:
        # square board
        ny = nx
    # initialize board
    xx, yy = np.meshgrid(
        np.linspace(-nx / 2, nx / 2, res),
        np.linspace(-ny / 2, ny / 2, res),
    )
    board = np.stack([xx, yy], axis=-1)
    board = torch.tensor(board, dtype=torch.float32) if to_torch_tensor else board
    return board


def rhombus_transform(rs, theta=60, degrees=True):
    """
    Assume rs are in rhombus basis. Then we can inversely transform those
    coordinates to the standard basis. This can be useful for e.g. meshing
    a rhombus and transforming the mesh to the standard basis afterwards.

    Parameters:
        rs (nsamples,2): array of 2d-vectors in cardinal basis to transform
        theta: (float) indicating the angle of the second basis vector e2
               relative to e1.
        degrees: (boolean) whether theta is in degrees or radians
    Returns:
        rs (nsamples,2): in rhombus coordinates
    """
    #rotmat_offset = rotation_matrix(-offset, degrees)
    rotmat = rotation_matrix(theta, degrees)
    #e1 = rotmat_offset @ np.array([1, 0])
    e1 = np.array([1, 0])
    e2 = rotmat @ e1
    # to rhombus coordinates
    basis_change = np.stack([e1, e2])
    # from rhombus coordinates to standard basis
    basis_change = np.linalg.inv(basis_change)
    return rs @ basis_change.T


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


def energy_statistic(X,Y):
    # extend to use geodesic distance rather than Euclidean
    all_to_all_fn = lambda X,Y: np.mean(np.linalg.norm(X[:,None]-Y[None],axis=-1))
    return 2*all_to_all_fn(X,Y) - all_to_all_fn(X,X) - all_to_all_fn(Y,Y)


def find_peaks(image):
    """
    Taken from cinpla/spatial-maps. But corrects center.

    Find peaks sorted by distance from center of image.
    Returns
    -------
    peaks : array
        coordinates for peaks in image as [row, column]
    """
    import scipy.ndimage as ndimage
    import scipy.ndimage.filters as filters
    image = image.copy()
    image[~np.isfinite(image)] = 0
    image_max = filters.maximum_filter(image, 3)
    is_maxima = (image == image_max)
    labels, num_objects = ndimage.label(is_maxima)
    indices = np.arange(1, num_objects+1)
    peaks = ndimage.maximum_position(image, labels=labels, index=indices)
    peaks = np.array(peaks)
    center = (np.array(image.shape)-1) / 2
    distances = np.linalg.norm(peaks - center, axis=1)
    peaks = peaks[distances.argsort()]
    return peaks