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

def intersect_vectorized(
    u1, v1, u2, v2, constraint1=[-np.inf, np.inf], constraint2=[-np.inf, np.inf]
):
    """
    Calculate intersection of two line segments defined as:
    l1 = {u1 + t1*v1 : u1,v1 in R^n, t1 in constraint1 subseq R},
    l2 = {u2 + t2*v2 : u2,v2 in R^n, t2 in constraint2 subseq R}
    Args:
        u1 (n,2) : bias of first line-segment
        v1 (n,2) : "slope" of first line-segment
        u2 (n,2) : bias of second line-segment
        v2 (n,2) : "slope" of first line-segment
        constraint1: 2d-array(like) of boundary points
                     for the "t-values" of the first line-segment
        constraint1: 2d-array(like) of boundary points
                     for the "t-values" of the second line-segment
    Returns:
        intersection (n,2) : point of intersection of the two line segments
        within_constraints (n,) : mask of which line segments intersects within constraints
    """
    matrix = np.stack([v1, -v2],axis=1)
    vector = u2 - u1
    try:
        solution = np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError as e:
        # Singular matrix (parallell line segments)
        print(e)
        return None, False
    
    within_constraints = (constraint1[0] <= solution[0] <= constraint1[1]) and (constraint2[0] <= solution[1] <= constraint2[1])
    return u1+solution[:,:1]*v1, within_constraints