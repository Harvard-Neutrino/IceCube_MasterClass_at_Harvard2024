import numpy as np
import matplotlib.pyplot as plt

def plot_heatmap(
    h: np.ndarray, bins: np.ndarray, **kwargs) -> None:

    figsize = (6,5)
    if "figsize" in kwargs.keys():
        figsize = kwargs["figsize"]
    fig, ax = plt.subplots(figsize=figsize)

    cmap = "cool_r"
    if "cmap" in kwargs.keys():
        cmap = kwargs["cmap"]

    extent = [np.min(bins), np.max(bins), np.min(bins), np.max(bins)]

    im = ax.imshow(h, extent=extent, cmap=cmap)
    ax.set_xlim(np.min(bins), np.max(bins))
    ax.set_ylim(np.min(bins), np.max(bins))
    ax.set_xlabel(r"$\sin\theta_{\rm{reco}}\left(\phi_{\rm{reco}} - \phi_{\rm{moon}}\right)$")
    ax.set_ylabel(r"$\theta_{\rm{reco}} - \theta_{\rm{moon}}$")
    cbar = plt.colorbar(im, label = r"$N_{\rm{evts}}$")

    if "figname" in kwargs.keys():
        plt.savefig(kwargs["figname"])

    plt.show()
