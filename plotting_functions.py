import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def barcode_plot(diagram,dims=2,norm_ax = 0):
    results = {}
    if norm_ax==0:
        largest_pers = 0
        for d in range(dims):
            results['h'+str(d)] = diagram[d]
            if np.max(diagram[d][np.isfinite(diagram[d])])>largest_pers:
                largest_pers = np.max(diagram[d][np.isfinite(diagram[d])])
    elif norm_ax!=0:
        largest_pers=norm_ax
    clrs = ['tab:blue','tab:orange','tab:green']#['b','r','g','m','c']
    diagram[0][~np.isfinite(diagram[0])] = largest_pers+0.1*largest_pers
    plot_prcnt = 0*np.ones(dims)
    to_plot = []
    for curr_h, cutoff in zip(diagram, plot_prcnt):
         bar_lens = curr_h[:,1] - curr_h[:,0]
         plot_h = curr_h[bar_lens >= np.percentile(bar_lens, cutoff)]
         to_plot.append(plot_h)
    fig = plt.figure(figsize=(3, 1.25))
    gs = gridspec.GridSpec(dims, 4)
    for curr_betti, curr_bar in enumerate(to_plot):
        ax = fig.add_subplot(gs[curr_betti, :])
        for i, interval in enumerate(reversed(curr_bar)):
            plt.plot([interval[0], interval[1]], [i, i], color=clrs[curr_betti],
                lw=1.5)
        if curr_betti == dims-1:
            ax.set_xlim([0, largest_pers+0.01])
            ax.set_ylim([-1, len(curr_bar)])
            ax.set_yticks([])
        else:
            ax.set_xlim([0, largest_pers+0.01])
            ax.set_xticks([])
            ax.set_ylim([-1, len(curr_bar)])
            ax.set_yticks([])


''' Draws circles around the points of a point cloud, first dimension contains the number of points '''

def rips_plot(pcloud,radius,graph=False,dmat=None,polygons=False,circles=True):
    plt.plot(pcloud[:,0],pcloud[:,1],'b.')
    fig = plt.gcf()
    ax = fig.gca()  
    for i in range(len(pcloud)):
        if circles == True:
            circle = plt.Circle((pcloud[i,0],pcloud[i,1]),radius,color='r',alpha = 0.025)
            ax.add_artist(circle)
        if graph == True:
            for j in range(len(pcloud)):
                if dmat[i,j]<=radius:
                    if i<j:
                        ax.plot([pcloud[i,0],pcloud[j,0]],[pcloud[i,1],pcloud[j,1]],'k',alpha=0.5)
                if polygons==True:
                    for k in range(len(pcloud)):
                        if dmat[i,j]<=radius and dmat[i,k]<=radius and dmat[j,k]<=radius:
                            polygon = Polygon(pcloud[[i,j,k],:])
                            p = PatchCollection([polygon],alpha=0.5)
                            p.set_array(np.array([5,50,100]))
                            ax.add_collection(p)
    return fig,ax
  
def annotate_imshow(D,round_val=2,txt_size=6):
    fig, ax = plt.subplots(1,1,dpi=200)
    ax.imshow(D,aspect='auto')
    for (j,i),label in np.ndenumerate(D):
        if label!=0:
            ax.text(i,j,round(label,round_val),ha='center',va='center',fontsize=txt_size)
