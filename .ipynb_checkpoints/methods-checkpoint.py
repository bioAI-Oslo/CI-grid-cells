import numpy as np


def rotation_matrix(theta, degrees=True):
    """
    2d-rotation matrix implementation
    """
    # convert to radians
    theta = theta * np.pi / 180 if degrees else theta
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))


def grid_cell(f, r0, gmax, orientation_offset=0, degrees=True):
    """
    Implementation following the logic described in the paper:
    From grid cells to place cells: A mathematical model
    - Moser & Einevoll

    Usage:
        Set static params for a grid cell as: gc = grid_cell(*args)
        Then calculate grid cell response wrt. pos=(x,y) as gc(pos)
    """
    rot_theta = 60  # degrees
    relative_R = rotation_matrix(rot_theta)
    init_R = rotation_matrix(orientation_offset, degrees)

    k1 = np.array([1.0, 0.0])  # init wave vector. unit length in x-direction
    k1 = init_R @ k1
    k2 = relative_R @ k1  # rotate k1 by 60degrees using R
    k3 = relative_R @ k2  # rotate k2 by 60degrees using R
    ks = np.array([k1, k2, k3])
    ks *= 2 * np.pi  # spatial angular frequency (unit-movement in space is one period)
    ks *= f  # user-defined spatial frequency

    def wrapper(r):
        if r.shape == (2,) and r0.shape == (2,):
            ws = (np.cos((r - r0[None, None]) @ ks.T) + 0.5) / 3
        ws = (np.cos((r - r0) @ ks.T) + 0.5) / 3
        ws = ws * gmax * 2 / 3
        return ws

    return wrapper


class GridModule:
    def __init__(
        self, f, r0, gmax=1, orientation_offset=0, approx_num_cells=10, degrees=True
    ):
        self.f = f
        self.radius = 1 / f
        self.gmax = gmax
        # multiply by fractional area between enclosing square and hexagon
        self.num_cells_in_square = (
            approx_num_cells * (2 * self.radius) ** 2 / hexagon_area(self.radius)
        )
        self.num_cells_in_square = round(self.num_cells_in_square)
        self.remap(r0, orientation_offset)

    def __call__(self, r):
        return np.sum(self.grid_cells(r), axis=-1)

    def remap(self, r, orientation_offset, degrees=True):
        self.hpoints = hexagon(self.radius, orientation_offset)
        xx, yy = np.meshgrid(
            np.linspace(-self.radius, self.radius, self.num_cells_in_square),
            np.linspace(-self.radius, self.radius, self.num_cells_in_square),
        )
        board = np.stack([xx, yy], axis=-1)
        mask = np.zeros(board.shape[:-1], dtype=bool)
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                mask[i, j] = is_in_hexagon(board[i, j], self.hpoints, np.zeros(2))

        self.phases = board + r
        self.phase_mask = mask
        self.orientation_offset = orientation_offset
        self.grid_cells = grid_cell(
            self.f,
            self.phases[:, :, None, None],
            self.gmax,
            self.orientation_offset,
            degrees,
        )


def hexagon(radius, orientation_offset=0):
    """
    Define the 6 vertices of a hexagon with chosen radius
    and orientation offset wrt. the cardinal x-axis. 
    """
    rotmat60 = rotation_matrix(60, degrees=True)
    rotmat_offset = rotation_matrix(orientation_offset, degrees=True)
    hpoints = np.array([radius, 0])  # start vector along cardinal x-axis
    hpoints = rotmat_offset @ hpoints
    hpoints = [hpoints]
    for _ in range(5):
        hpoints.append(rotmat60 @ hpoints[-1])
    return np.array(hpoints)


def is_in_hexagon(point, hpoints, center):
    """
    Check if a 2d-point is within a hexagon defined by its 6 
    points 'hpoints' with phase 'center'.
    """
    u2, v2 = center, point - center
    hpoints = hpoints + center
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


def hexagon_area(radius):
    return 3 * np.sqrt(3) * radius * radius / 2


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
