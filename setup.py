import numpy as np

def mnras_size(fig_width_pt, square=False):
    inches_per_pt = 1.0 / 72.00  # Convert pt to inches
    golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Most aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # Figure width in inches
    if square:
        fig_height = fig_width
    else:
        fig_height = fig_width * golden_mean
    return [fig_width, fig_height]
  
red = "#e41a1c"
blue = "#377eb8"
green = "#4daf4a"
purple = "#984ea3"
orange = "#ff7f00"

overleaf_path = (
    "/Users/daniel/Dropbox (Sydney Uni)/Apps/Overleaf/AF exoplanet/figures/"
)

