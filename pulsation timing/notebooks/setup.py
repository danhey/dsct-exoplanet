import numpy as np
import corner
import pandas as pd
import matplotlib.pyplot as plt
import exoplanet as xo
import pymc3 as pm
from maelstrom import Maelstrom
from astropy.stats import LombScargle
from astropy.convolution import convolve, Box1DKernel
import math
import matplotlib
from lightkurve import search_lightcurvefile
import lightkurve as lk
from scipy.ndimage import gaussian_filter
import glob
from maelstrom.utils import amplitude_spectrum, mass_function
import pickle # python3
import astropy.units as u
from astropy.stats import BoxLeastSquares

def mnras_size(fig_width_pt, square=False):
    inches_per_pt = 1.0 / 72.00  # Convert pt to inches
    golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Most aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # Figure width in inches
    if square:
        fig_height = fig_width
    else:
        fig_height = fig_width * golden_mean
    return [fig_width, fig_height]

def high_pass(t, y, width=3.):
    y_low = gaussian_filter(y, width)
    return y - y_low

def get_kepler_lc(kic_id):
    file = glob.glob(f'../data/lightcurves/Kepler/*{kic_id}.txt')[0]
    t, y = np.loadtxt(file, usecols=(0,1)).T
    return t, y
    
def preprocess_lc(t, y):
    y = high_pass(t, y)
    lc = lk.LightCurve(t, y).remove_outliers()
    return lc.time, lc.flux


red = "#e41a1c"
blue = "#377eb8"
green = "#4daf4a"
purple = "#984ea3"
orange = "#ff7f00"

overleaf_path = (
    "/Users/daniel/Dropbox (Sydney Uni)/Apps/Overleaf/PM planet/figs/"
)

# import matplotlib as mpl
# mpl.rcParams['xtick.labelsize'] = 8.5
# mpl.rcParams['ytick.labelsize'] = 8.5
# mpl.rcParams['axes.labelsize'] = 8.5
# # mpl.rcParams['figure.figsize'] = (10,7)
# mpl.rcParams["figure.dpi"] = 100
# mpl.rcParams["savefig.dpi"] = 300

# from matplotlib import rc
# from matplotlib import cm
# rc('font', **{'serif': ['Computer Modern']})
# rc('text', usetex=False)



# koi_dsct = np.array([3965201, 5202905, 5617259, 6032730, 6116172, 6670742, 9111849, 9289704, 9775385, 11013201], dtype=np.int64)
# toi_dsct = np.array([156987351,
# 193413306,
# 202563254,
# 255704097,
# 295599256,
# 372913430,
# 80275202,
# 89759617,
# 120269103,
# 409934330,])
