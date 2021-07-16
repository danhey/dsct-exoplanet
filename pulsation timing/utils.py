import numpy as np
from astropy.timeseries import LombScargle
from scipy import optimize
from scipy.ndimage import gaussian_filter
import glob
import lightkurve as lk

def estimate_frequencies(
    x, y, fmin=None, fmax=None, max_peaks=3, oversample=4.0, optimize_freq=True
):
    tmax = x.max()
    tmin = x.min()
    dt = np.median(np.diff(x))
    df = 1.0 / (tmax - tmin)
    ny = 0.5 / dt
    if fmin is None:
        fmin = df
    if fmax is None:
        fmax = 2*ny
    freq = np.arange(fmin, fmax, df / oversample)
    power = LombScargle(x, y).power(freq)
    # Find peaks
    peak_inds = (power[1:-1] > power[:-2]) & (power[1:-1] > power[2:])
    peak_inds = np.arange(1, len(power) - 1)[peak_inds]
    peak_inds = peak_inds[np.argsort(power[peak_inds])][::-1]
    peaks = []
    for j in range(max_peaks):
        i = peak_inds[0]
        freq0 = freq[i]
        alias = 2.0 * ny - freq0
        m = np.abs(freq[peak_inds] - alias) > 25 * df
        m &= np.abs(freq[peak_inds] - freq0) > 25 * df
        peak_inds = peak_inds[m]
        peaks.append(freq0)
    peaks = np.array(peaks)
    if optimize_freq:
        def chi2(nu):
            arg = 2 * np.pi * nu[None, :] * x[:, None]
            D = np.concatenate([np.cos(arg), np.sin(arg), np.ones((len(x), 1))], axis=1)
            # Solve for the amplitudes and phases of the oscillations
            DTD = np.matmul(D.T, D)
            DTy = np.matmul(D.T, y[:, None])
            w = np.linalg.solve(DTD, DTy)
            model = np.squeeze(np.matmul(D, w))
            chi2_val = np.sum(np.square(y - model))
            return chi2_val
        res = optimize.minimize(chi2, [peaks], method="L-BFGS-B")
        return res.x
    else:
        return peaks
    
    
def amplitude_spectrum(t, y, fmin=None, fmax=None, freq=None, oversample_factor=10.0):
    """ 
    Calculates the amplitude spectrum of a given signal
    
    Parameters
    ----------
        t : `array`
            Time values 
        y : `array`
            Flux or magnitude measurements
        fmin : float (default None)
            Minimum frequency to calculate spectrum. Defaults to df
        fmax : float
            Maximum frequency to calculate spectrum. Defaults to Nyquist.
        oversample_factor : float
            Amount by which to oversample the spectrum. Defaults to 10.
    """
    tmax = t.max()
    tmin = t.min()
    df = 1.0 / (tmax - tmin)

    if fmin is None:
        fmin = df
    if fmax is None:
        fmax = 0.5 / np.median(np.diff(t))  # *nyq_mult
    if freq is None:
        freq = np.arange(fmin, fmax, df / oversample_factor)
    model = LombScargle(t, y)
    sc = model.power(freq, method="fast", normalization="psd")

    fct = np.sqrt(4.0 / len(t))
    amp = np.sqrt(sc) * fct

    return freq, amp


def high_pass(t, y, width=3.):
    y_low = gaussian_filter(y, width)
    return y - y_low

# def get_kepler_lc(kic_id):
#     file = glob.glob(f'data/lightcurves/Kepler/*{kic_id}.txt')[0]
#     t, y = np.loadtxt(file, usecols=(0,1)).T
#     return t, y

def get_kepler_lc(kic_id):
    file = glob.glob(f'../data/lightcurves/PDC/*{kic_id}.csv')[0]
    t, y = np.loadtxt(file, usecols=(0,1)).T
    return t, y
    
def preprocess_lc(t, y):
    y = high_pass(t, y)
    lc = lk.LightCurve(t, y).remove_outliers().remove_nans()
    return lc.time.value, lc.flux.value