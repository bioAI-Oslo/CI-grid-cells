import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import seaborn as sns


def project_cmap():
    """
    Custom color map
    """
    # Markus' dream color
    # return sns.color_palette('Blues',as_cmap=True)
    return sns.cubehelix_palette(
        start=-0.25, rot=0, gamma=0.75, hue=2, light=1, dark=0.1, as_cmap=True
    )


def scatter3d(
    data, ncols=4, nrows=4, fig=None, axs=None, azim_elev_title=False, **kwargs
):
    assert data.shape[-1] == 3, "data must have three axes. No more, no less."
    if data.ndim > 2:
        data = data.reshape(-1, 3)
    if fig is None or axs is None:
        fig, axs = plt.subplots(
            ncols=ncols, nrows=nrows, subplot_kw={"projection": "3d"}
        )
    else:
        nrows, ncols = axs.shape
    azims = np.linspace(0, 180, ncols + 1)[:-1]
    elevs = np.linspace(0, 90, nrows + 1)[:-1]
    view_angles = np.stack(np.meshgrid(azims, elevs), axis=-1)
    grid = np.meshgrid(np.linspace(0,1,nrows,endpoint=False), np.linspace(0,1,ncols,endpoint=False))
    grid = np.stack(grid, axis=-1)
    for i in range(nrows):
        for j in range(ncols):
            axs[i,j].scatter(xs=data[:, 0], ys=data[:, 1], zs=data[:, 2], **kwargs)
            axs[i,j].azim = view_angles[i, j, 0]
            axs[i,j].elev = view_angles[i, j, 1]
            axs[i,j].axis("off")
            if azim_elev_title:
                axs[i,j].set_title(f"azim={axs[i,j].azim}, elev={axs[i,j].elev}")
            axs[i,j].set_position([*grid[i,j], 1/ncols, 1/nrows])
    return fig, axs

def scatter3d2(
    data, ncols=4, nrows=4, fig=None, axs=None, azim_elev_title=False, **kwargs
):
    assert data.shape[-1] == 3, "data must have three axes. No more, no less."
    if data.ndim > 2:
        data = data.reshape(-1, 3)
    if fig is None or axs is None:
        fig, axs = plt.subplots(
            ncols=ncols, nrows=nrows, subplot_kw={"projection": "3d"}
        )
    else:
        nrows, ncols = axs.shape
    azims = np.linspace(0, 180, ncols + 1)[:-1]
    elevs = np.linspace(0, 90, nrows + 1)[:-1]
    view_angles = np.stack(np.meshgrid(azims, elevs), axis=-1)
    grid = np.meshgrid(np.linspace(0,1,nrows), np.linspace(0,1,ncols))
    grid = np.stack(grid, axis=-1).reshape(-1, 2)
    for i in range(nrows):
        for j in range(ncols):
            axs[i,j].scatter(xs=data[:, 0], ys=data[:, 1], zs=data[:, 2], **kwargs)
            axs[i,j].azim = view_angles[i, j, 0]
            axs[i,j].elev = view_angles[i, j, 1]
            axs[i,j].axis("off")
            if azim_elev_title:
                axs[i,j].set_title(f"azim={axs[i,j].azim}, elev={axs[i,j].elev}")
            axs[i,j].set_position([grid[i,j,0], grid[i,j,1], 1/ncols, 1/nrows])


    for i, ax in enumerate(axs.flat):
        ax.scatter(xs=data[:, 0], ys=data[:, 1], zs=data[:, 2], **kwargs)
        ax.azim = view_angles[i, 0]
        ax.elev = view_angles[i, 1]
        ax.axis("off")
        if azim_elev_title:
            ax.set_title(f"azim={ax.azim}, elev={ax.elev}")
        ax.set_position()
        ax.set_position([0, 0, 1, 1])
    return fig, axs


def plot_samples_and_tiling(gridmodule, ratemaps, ratemap_examples=0, **kwargs):
    fig, axs = plt.subplots(ncols=2 + ratemap_examples, **kwargs)
    gridmodule.plot(fig, axs[0])
    axs[0].scatter(*gridmodule.phase_offsets.T, s=5, color="orange", zorder=2)
    axs[0].axis("off")

    for i, ratemap in enumerate(ratemaps[:ratemap_examples]):
        axs[i + 1].imshow(ratemap, origin="lower")
        axs[i + 1].axis("off")

    axs[-1].imshow(np.around(np.sum(ratemaps, axis=0), decimals=10), origin="lower")
    axs[-1].axis("off")
    # fig.savefig(fname)
    return fig, axs


def barcode_plot(diagram, dims=2, norm_ax=0, fig=None):
    results = {}
    if norm_ax == 0:
        largest_pers = 0
        for d in range(dims):
            results["h" + str(d)] = diagram[d]
            if np.max(diagram[d][np.isfinite(diagram[d])]) > largest_pers:
                largest_pers = np.max(diagram[d][np.isfinite(diagram[d])])
    elif norm_ax != 0:
        largest_pers = norm_ax
    clrs = ["tab:blue", "tab:orange", "tab:green"]  # ['b','r','g','m','c']
    diagram[0][~np.isfinite(diagram[0])] = largest_pers + 0.1 * largest_pers
    plot_prcnt = 0 * np.ones(dims)
    to_plot = []
    for curr_h, cutoff in zip(diagram, plot_prcnt):
        bar_lens = curr_h[:, 1] - curr_h[:, 0]
        plot_h = curr_h[bar_lens >= np.percentile(bar_lens, cutoff)]
        to_plot.append(plot_h)
    fig = plt.figure(figsize=(2, 1.25/3*dims)) if fig is None else fig
    gs = gridspec.GridSpec(dims, 4)
    for curr_betti, curr_bar in enumerate(to_plot):
        ax = fig.add_subplot(gs[curr_betti, :])
        for i, interval in enumerate(reversed(curr_bar)):
            plt.plot([interval[0], interval[1]], [i, i], color=clrs[curr_betti], lw=1.5)
        ax.set_ylabel(f"H$^{curr_betti}$", rotation=0, labelpad=10)
        if curr_betti == dims - 1:
            ax.set_xlim([0, largest_pers + 0.01])
            ax.set_ylim([-0.15*len(curr_bar), len(curr_bar)])
            ax.set_yticks([])
        else:
            ax.set_xlim([0, largest_pers + 0.01])
            ax.set_xticks([])
            ax.set_ylim([-0.15*len(curr_bar), len(curr_bar)])
            ax.set_yticks([])
    return fig, ax


""" Draws circles around the points of a point cloud, first dimension contains the number of points """


def rips_plot(pcloud, radius, graph=False, dmat=None, polygons=False, circles=True):
    plt.plot(pcloud[:, 0], pcloud[:, 1], "b.")
    fig = plt.gcf()
    ax = fig.gca()
    for i in range(len(pcloud)):
        if circles == True:
            circle = plt.Circle(
                (pcloud[i, 0], pcloud[i, 1]), radius, color="r", alpha=0.025
            )
            ax.add_artist(circle)
        if graph == True:
            for j in range(len(pcloud)):
                if dmat[i, j] <= radius:
                    if i < j:
                        ax.plot(
                            [pcloud[i, 0], pcloud[j, 0]],
                            [pcloud[i, 1], pcloud[j, 1]],
                            "k",
                            alpha=0.5,
                        )
                if polygons == True:
                    for k in range(len(pcloud)):
                        if (
                            dmat[i, j] <= radius
                            and dmat[i, k] <= radius
                            and dmat[j, k] <= radius
                        ):
                            polygon = Polygon(pcloud[[i, j, k], :])
                            p = PatchCollection([polygon], alpha=0.5)
                            p.set_array(np.array([5, 50, 100]))
                            ax.add_collection(p)
    return fig, ax


def minimal_ticks(ax):
    ax.set_xticks([ax.get_xticks()[1], ax.get_xticks()[-1]])
    ax.set_yticks([ax.get_yticks()[1], ax.get_yticks()[-1]])
    return ax


def set_size(width=345.0, fraction=1, mode="wide"):
    """Set figure dimensions to avoid scaling in LaTeX.
    Taken from:
    https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    To get the width of a latex document, print it with:
    \the\textwidth
    (https://tex.stackexchange.com/questions/39383/determine-text-width)
    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    mode: str
            Whether figure should be scaled by the golden ratio in height
            or width
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width  # * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    if mode == "wide":
        fig_height_in = fig_width_in * golden_ratio
    elif mode == "tall":
        fig_height_in = fig_width_in / golden_ratio
    elif mode == "square":
        fig_height_in = fig_width_in
    fig_height_max = 550.0 / 72.27
    if mode == "max" or fig_height_in > fig_height_max:
        # standard max-height of latex document
        fig_height_in = fig_height_max
    fig_dim = (fig_width_in, fig_height_in)
    if isinstance(fraction, (int, float)):
        fraction = (fraction, fraction)
    fig_dim = (fig_width_in * fraction[0], fig_height_in * fraction[1])
    return fig_dim


def annotate_imshow(D, round_val=2, txt_size=6,cmap='coolwarm',**kwargs):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(D, aspect="auto",cmap=cmap,**kwargs)
    for (j, i), label in np.ndenumerate(D):
        if label != 0:
            ax.text(
                i,
                j,
                round(label, round_val),
                ha="center",
                va="center",
                fontsize=txt_size,
            )

def colorbar_axis(fig, ax, im):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')


def multiimshow(zz, figsize=(1,1), normalize=True, add_colorbar=True, rect=(0,0,1,0.87), axes_pad=0.05, **kwargs):
    # prepare figure
    ncols = int(np.ceil(np.sqrt(zz.shape[0])))
    nrows = int(round(np.sqrt(zz.shape[0])))
    from mpl_toolkits.axes_grid1 import ImageGrid
    fig = plt.figure(figsize=figsize)
    if add_colorbar and normalize:
        grid = ImageGrid(fig, rect=rect, nrows_ncols=(nrows, ncols), axes_pad=axes_pad, cbar_mode='single', cbar_location='right', cbar_pad=0.1, cbar_size='5%')
    else:
        grid = ImageGrid(fig, rect=rect, nrows_ncols=(nrows, ncols), axes_pad=axes_pad)
    vmin, vmax = (np.nanmin(zz), np.nanmax(zz)) if normalize else (None, None)
    # plot response maps using imshow
    for ax, data in zip(grid, zz):
        im = ax.imshow(data, vmin=vmin, vmax=vmax, **kwargs)
    [ax.axis('off') for ax in grid]
    fig.colorbar(im, cax=grid.cbar_axes[0]) if (normalize and add_colorbar) else None
    return fig, grid.axes_all