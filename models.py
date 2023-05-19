import torch
import numpy as np
import copy

from methods import HexagonalGCs
from torch_topological.nn import VietorisRipsComplex


class LinDecoder(HexagonalGCs):
    def __init__(self, hex_metric=False, least_squares=False, pos_scale=1, **kwargs):
        super(LinDecoder, self).__init__(**kwargs)

        # Parameters
        self.pos_scale = pos_scale
        self.least_squares = least_squares
        if not least_squares:
            self.xyweights = torch.nn.Parameter(
                torch.ones(self.ncells, 2) / (self.ncells * 2), requires_grad=True
            )
        else:
            self.xyweights = torch.ones(self.ncells, 2) / (self.ncells * 2)
        self.hex_metric = hex_metric
        if hex_metric:
            diam = self.unit_cell.basis[1, 1] * 2  #
            a = torch.tensor([[0.0, 0.0]])
            b = torch.tensor([[-0.5, np.sqrt(3.0) / 2]]) * diam
            c = torch.tensor([[-0.5, -np.sqrt(3.0) / 2]]) * diam
            d = torch.tensor([[0.5, np.sqrt(3.0) / 2]]) * diam
            e = torch.tensor([[0.5, -np.sqrt(3.0) / 2]]) * diam
            f = torch.tensor([[-1.0, 0.0]]) * diam
            g = torch.tensor([[1.0, 0.0]]) * diam
            self.addition = torch.flip(
                torch.concatenate((a, b, c, d, e, f, g), 0), (1, 0)
            )

    def loss_fn(self, pos):
        pos *= self.pos_scale
        activity = self(pos)
        if self.least_squares:
            self.xyweights = torch.linalg.lstsq(activity, pos).solution
        #            self.xyweights = torch.linalg.pinv(activity) @ pos
        decode_pos = torch.matmul(activity, self.xyweights)
        if self.hex_metric:
            diffall = torch.zeros(7, len(pos))
            for i in range(7):
                diffall[i] = torch.sum(
                    torch.square((decode_pos + self.addition[i]) - pos), 1
                )
            return torch.sum(torch.min(diffall, 0).values)
        else:
            return torch.sum(torch.square(decode_pos - pos))

    def loss_fn2(self, pos):
        pos *= self.pos_scale
        activity = self(pos)
        xyweights = torch.linalg.pinv(activity) @ pos
        decode_pos = torch.matmul(activity, xyweights)
        return torch.sum(torch.square(decode_pos - pos))


class Similitude(HexagonalGCs):
    def __init__(self, **kwargs):
        super(Similitude, self).__init__(**kwargs)

    def loss_fn(self, r):
        J = self.jacobian(r)
        det_J = self.the_jacobian(J)
        return torch.var(det_J)


class Homology(HexagonalGCs):
    def __init__(self, **kwargs):
        super(Homology, self).__init__(**kwargs)

    def loss_fn(self, r, p=2):
        out = self(r)
        dist = torch.cdist(out, out)
        hom = VietorisRipsComplex(dim=2)(dist, treat_as_distances=True)
        pers1 = torch.cat(
            [torch.zeros(3), torch.sort(hom[1][1][:, 1] - hom[1][1][:, 0])[0]]
        )
        pers2 = torch.cat(
            [torch.zeros(3), torch.sort(hom[2][1][:, 1] - hom[2][1][:, 0])[0]]
        )
        loss = (
            -torch.sum(pers1[-2:] ** p)
            - pers2[-1] ** p
            + torch.sum(pers1[:-2] ** p)
            + torch.sum(pers2[:-1] ** p)
        )
        return loss


class JitterCI(HexagonalGCs):
    def __init__(self, r_magnitude=0.01, p_magnitude=0.01, **kwargs):
        super(JitterCI, self).__init__(**kwargs)
        self.r_magnitude, self.p_magnitude = r_magnitude, p_magnitude

    def s(self, r, dr, dp):
        """
        s-function as defined in Xu2022 to learn conformal isometry

        Parameters:
            r (nsamples,2): np.ndarray matrix of 2D-spatial positions
            dr (nsamples,2): np.ndarray matrix of 2D-spatial jitter positions
            dp (ncells,2): np.ndarray matrix of 2D-spatial jitter phase-positions
        Returns:
            s-function evaluated (nsamples
        """
        # direct forward
        f = self(r)
        df = self(r + dr, dp)
        # rescale on outer product
        rescale_r = torch.sum(dr**2, axis=-1)
        rescale_p = torch.sum(dp**2, axis=-1)
        rescale_rp = rescale_r[:, None] + rescale_p[None]
        return 2 * torch.sum((f - df) ** 2, axis=-1) / torch.sum(rescale_rp, axis=-1)

    def loss_fn(self, r):
        """
        Conformal isometry loss using jittering. Follows the formulation in
        Xu2022 as this avoids the need for also finding/learning the conformal
        isometry scale directly.
        Additionally adds a robustness (wrt. the parameters, i.e. phases) term.
        This is achieved by also jittering the parameters.
        """
        # sample perturbations for input and parameters
        dr1, magnitudes_space = self.jitter(r.shape[0], magnitude=self.r_magnitude)
        dr2, _ = self.jitter(r.shape[0], magnitudes_space)
        dp1, magnitudes_phases = self.jitter(
            self.phases.shape[0], magnitude=self.p_magnitude
        )
        dp2, _ = self.jitter(self.phases.shape[0], magnitudes_phases)
        # dp1 = torch.normal(0, self.p_magnitude, size=(r.shape[0], self.ncells), dtype=self.dtype)
        # dp2 = torch.normal(0, self.p_magnitude, size=(r.shape[0], self.ncells), dtype=self.dtype)
        # perturb parameters and inputs
        s1 = self.s(r, dr1, dp1)
        s2 = self.s(r, dr2, dp2)
        loss = torch.mean((s1 - s2) ** 2)
        return loss


class JacobianCI(HexagonalGCs):
    def __init__(self, scale=None, p_magnitude=0, **kwargs):
        super(JacobianCI, self).__init__(**kwargs)
        self.p_magnitude = p_magnitude
        # scale of similitude
        self.set_scale(scale)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def set_scale(self, scale=None):
        if scale is None:
            # conformally isometric scaling LAW
            scale = self.ncells * 1.4621597785714284
        self.scale = torch.nn.Parameter(
            torch.tensor(scale, dtype=self.dtype), requires_grad=True
        )
        return self.scale

    def loss_fn(self, r):
        dp, _ = (
            self.jitter(self.phases.shape[0], magnitude=self.p_magnitude)
            if self.p_magnitude
            else (None, None)
        )
        # dp = torch.normal(0, self.p_magnitude, size=(r.shape[0], self.ncells), dtype=self.dtype)
        J = self.jacobian(r, dp)
        # (nsamples,2,2)
        metric_tensor = self.metric_tensor(J)
        diag_elems = torch.diagonal(metric_tensor, dim1=-2, dim2=-1)
        lower_triangular_elems = torch.tril(metric_tensor, diagonal=-1)
        loss = torch.sum((diag_elems - self.scale) ** 2, dim=-1) + 2 * torch.sum(
            lower_triangular_elems**2, dim=(-2, -1)
        )
        return torch.mean(loss)


class PlaceCells(HexagonalGCs):
    """
    torch model for learning optimal place cell phases
    """

    def __init__(
        self,
        ncells=3,
        f=1,
        init_rot=0,
        sig=1,
        scale=None,
        dtype=torch.float32,
        **kwargs
    ):
        super(PlaceCells, self).__init__(ncells, f, init_rot, **kwargs)

        self.sig = sig  # place cell tuning width
        # conformal scaling factor.
        if scale is None:
            self.scale = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
        else:
            self.scale = scale

    def forward(self, r, rectify=False):
        """
        Parameters:
            r (nsamples,2): spatial samples
        Returns:
            activity (nsamples,ncells): activity of all cells on spatial samples
        """
        activity = torch.exp(
            -torch.sum(
                (r[:, None] - self.phases[None]) ** 2 / (2 * self.sig**2), dim=-1
            )
        )
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
        J = -1 / self.sig**2 * (r[:, None] - self.phases[None]) * p[..., None]
        return J

    def metric(self, r):
        J = self.jacobian(r)
        return torch.transpose(J, -2, -1) @ J

    def loss_fn(self, r):
        """
        Scaled conformal Isometry loss
        """
        g = self.metric(r)
        diag_loss = (g[:, 0, 0] - self.scale * 100) ** 2 + (
            g[:, 1, 1] - self.scale * 100
        ) ** 2
        cross_loss = 2 * g[:, 0, 1] ** 2
        return torch.mean(diag_loss + cross_loss)


class Similitude3(HexagonalGCs):
    def __init__(self, scale=None, shift=-1, **kwargs):
        super(Similitude3, self).__init__(**kwargs)
        # scale of similitude
        if scale is None:
            # conformally isometric scaling LAW
            scale = self.ncells * 0.014621597785714284
        self.scale = torch.nn.Parameter(
            torch.tensor(scale, dtype=torch.float32), requires_grad=True
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.shift = shift

    def forward(self, r):
        jittered_phases = self.phases + r[-1]
        activity = torch.cos((r[0][:, None] - jittered_phases[None]) @ self.ks.T)
        activity = torch.sum(activity, dim=-1)  # sum plane waves
        activity = 2 / 9 * (1 - self.shift) * activity + 1 / 3 * (1 + 2 * self.shift)
        activity = self.relu(activity) if self.relu else activity
        return activity

    def jacobian(self, r):
        """
        Jacobian of the forward function
        """
        jittered_phases = self.phases + r[-1]
        J_tmp = (
            -2
            / 9
            * (1 - self.shift)
            * torch.sin((r[0][:, None] - jittered_phases[None]) @ self.ks.T)
        )

        Jx = torch.sum(J_tmp * self.ks[:, 0], dim=-1)
        Jy = torch.sum(J_tmp * self.ks[:, 1], dim=-1)

        J = torch.stack([Jx, Jy], dim=-1)
        if self.relu:
            relu_grad_mask = self.forward(r) > 0
            J = relu_grad_mask[..., None] * J
        return J

    def metric_metrics(self, r, keepdims=False):
        J = self.jacobian(r)
        metric_tensor = self.metric_tensor(J)

        g11 = metric_tensor[..., 0, 0]
        g22 = metric_tensor[..., 1, 1]

        off_diag = metric_tensor[..., 1, 0]
        return g11, g22, off_diag

    def loss_fn(self, r, keepdims=False):
        ra = r[0]
        rb = ra + r[1]
        rc = ra + r[2]
        dr = r[3]

        ga = self((ra, r[-1]))
        gb = self((rb, r[-1]))
        gc = self((rc, r[-1]))

        sab = torch.sum((ga - gb) ** 2, dim=-1) / dr**2
        sac = torch.sum((ga - gc) ** 2, dim=-1) / dr**2

        loss = (sab - sac) ** 2

        if not keepdims:
            return torch.mean(loss)
        else:
            return loss

    def metric_loss(self, r, keepdims=False):
        g11, g22, off_diag = self.metric_metrics((r[0], r[-1]))

        loss = (
            torch.var(g11)
            + torch.var(g22)
            + torch.mean((g11 - g22) ** 2)
            + 2 * torch.mean(off_diag**2)
        )

        if not keepdims:
            return torch.mean(loss)
        else:
            return loss
