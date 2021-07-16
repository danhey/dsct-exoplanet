import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares, LombScargle
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import glob

def harmonic_prewhiten(time, flux, frequencies, amplitudes):
    
    def model(time, *theta):
        freq, amp, phi = np.reshape(theta, (3, len(frequencies)))
        return np.sum(amp[:,None] * np.cos((2 * np.pi * freq[:,None] * (time - 0.)) + phi[:,None]), axis=0)
    
    phases = np.ones(len(frequencies))
    x0 = np.array([frequencies, amplitudes, phases]).flatten()
    popt, _ = curve_fit(model, time, flux, p0=x0)
    return flux - model(time, popt), popt

def preprocess_lc(x, y, yerr, ax=None):
    mu = np.median(y)
    y = (y / mu - 1) * 1e3
    yerr = yerr * 1e3
    
    # Identify outliers
    m = np.ones(len(y), dtype=bool)
    for i in range(10):
        y_prime = np.interp(x, x[m], y[m])
        smooth = savgol_filter(y_prime, 101, polyorder=3)
        resid = y - smooth
        sigma = np.sqrt(np.mean(resid ** 2))
        m0 = np.abs(resid) < 3 * sigma
        if m.sum() == m0.sum():
            m = m0
            break
        m = m0

    # Only discard positive outliers
    m = resid < 3 * sigma
    
    if ax is not None:
        # Plot the data
        plt.plot(x, y, "k", label="data")
        plt.plot(x, smooth)
        plt.plot(x[~m], y[~m], "xr", label="outliers")
        plt.legend()
        plt.xlim(x.min(), x.max())
        plt.xlabel("Time")
        plt.ylabel("Flux [ppt]")
        
    
    # Make sure that the data type is consistent
    x = np.ascontiguousarray(x[m], dtype=np.float64)
    y = np.ascontiguousarray(y[m], dtype=np.float64)
    yerr = np.ascontiguousarray(yerr[m], dtype=np.float64)
    smooth = np.ascontiguousarray(smooth[m], dtype=np.float64)
    return x, y, yerr, smooth

def estimate_background(x, y, log_width=0.01):
    count = np.zeros(len(x), dtype=int)
    bkg = np.zeros_like(x)
    x0 = np.log10(x[0])
    while x0 < np.log10(x[-1]):
        m = np.abs(np.log10(x) - x0) < log_width
        bkg[m] += np.median(y[m])
        count[m] += 1
        x0 += 0.5 * log_width
    return bkg / count

def amplitude_spectrum(t, y, fmin=None, fmax=None, oversample_factor=10.0):
    
    tmax = t.max()
    tmin = t.min()
    df = 1.0 / (tmax - tmin)

    if fmin is None:
        fmin = df
    if fmax is None:
        fmax = 0.5 / np.median(np.diff(t))  # *nyq_mult

    freq = np.arange(fmin, fmax, df / oversample_factor)
    model = LombScargle(t, y)
    sc = model.power(freq, method="fast", normalization="psd")

    fct = np.sqrt(4.0 / len(t))
    amp = np.sqrt(sc) * fct

    return freq, amp

def plot_river(time, flux, period, t0, ax, cmap='Blues_r'):
    s = np.argsort(time)
    x, y, e = time[s], flux[s], (np.ones_like(flux) * np.nan)[s]
    med = np.nanmedian(flux)
    e /= med
    y /= med

    bin_points=1
    minimum_phase=-0.5
    maximum_phase=0.5
    n = int(period/np.nanmedian(np.diff(x)) * (maximum_phase - minimum_phase)/bin_points)
    bin_func = lambda y, e: (np.nanmedian(y), np.nansum(e**2)**0.5/len(e))
    
    ph = x/period % 1
    cyc = np.asarray((x - x % period)/period, int)
    cyc -= np.min(cyc)

    phase = (t0 % period) / period
    ph = ((x - (phase * period)) / period) % 1
    cyc = np.asarray((x - ((x - phase * period) % period))/period, int)
    cyc -= np.min(cyc)
    ph[ph > 0.5] -= 1

    ar = np.empty((n, np.max(cyc) + 1))
    ar[:] = np.nan
    bs = np.linspace(minimum_phase, maximum_phase, n)
    cycs = np.arange(0, np.max(cyc) + 1)

    ph_masks = [(ph > bs[jdx]) & (ph <= bs[jdx+1]) for jdx in range(n-1)]
    qual_mask = np.isfinite(y)
    for cyc1 in np.unique(cyc):
        cyc_mask = cyc == cyc1
        if not np.any(cyc_mask):
            continue    
        for jdx, ph_mask in enumerate(ph_masks):
            if not np.any(cyc_mask & ph_mask & qual_mask):
                ar[jdx, cyc1] = np.nan
            else:
                ar[jdx, cyc1] = bin_func(y[cyc_mask & ph_mask],
                                         e[cyc_mask & ph_mask])[0]
                
    ar *= np.nanmedian(flux)
    d = np.max([np.abs(np.nanmedian(ar) - np.nanpercentile(ar, 5)),
                    np.abs(np.nanmedian(ar) - np.nanpercentile(ar, 95))])
    vmin = np.nanmedian(ar) - d
    vmax = np.nanmedian(ar) + d
    
    im = ax.pcolormesh(bs, cycs, ar.T, 
                       vmin=vmin, vmax=vmax, 
                       cmap=cmap)
    cbar = plt.colorbar(im, ax=ax)
    ax.set_ylim(cyc.max(), 0)
    return ax

# def deep_clean(x, y, snr=4, log_width=0.05, maxiter=100):
#     nyq = 0.5 / np.median(np.diff(x))
#     df = 1.0 / (x.max() - x.min())
#     fmin, fmax = df, 2*nyq

#     f,a = amplitude_spectrum(x, y, fmin=fmin, fmax=fmax)
#     noise = estimate_background(f,a, log_width=log_width)
#     a_snr = a/noise
#     idx = np.nanargmax(a)
#     f0, a0 = f[idx], a[idx]
    
#     pre_res = []
#     while ((a_snr)[idx] > snr) & (len(pre_res) < maxiter):
#         y, res = harmonic_prewhiten(x, y, [f0], [a0])
#         pre_res.append(res)
#         f, a = amplitude_spectrum(x, y, fmin=fmin, fmax=fmax)
#         idx = np.nanargmax(a)
#         f0, a0 = f[idx], a[idx]

#     pre_res = np.array(pre_res)
#     if len(pre_res) > 0:
#         m = pre_res[:,1] < 0
#         if len(m) > 0:
#             pre_res[:,1][m] *= -1.
#             pre_res[:,2][m] += np.pi
    
#     return x, y, pre_res

def deep_clean(x, y, snr=4, log_width=0.05, maxiter=100):
    nyq = 0.5 / np.median(np.diff(x))
    df = 1.0 / (x.max() - x.min())
    fmin, fmax = df, 2*nyq

    f,a = amplitude_spectrum(x, y, fmin=fmin, fmax=fmax)
    noise = estimate_background(f,a, log_width=log_width)
    a_snr = a/noise
    m = (f > 1) & (f < (2*nyq - 1))
    idx = np.nanargmax(a[m])
    f0, a0 = f[m][idx], a[m][idx]
    
    pre_res = []
    while ((a_snr)[m][idx] > snr) & (len(pre_res) < maxiter):
        y, res = harmonic_prewhiten(x, y, [f0], [a0])
        pre_res.append(res)
        f, a = amplitude_spectrum(x, y, fmin=fmin, fmax=fmax)
        m = (f > 1) & (f < (2*nyq - 1))
        idx = np.nanargmax(a[m])
        if a_snr[m][idx] > snr:
            f0, a0 = f[m][idx], a[m][idx]
        else:
            m = np.ones_like(a_snr, dtype=bool)
            idx = np.nanargmax(a[m])
            f0, a0 = f[idx], a[idx]

    pre_res = np.array(pre_res)
    if len(pre_res) > 0:
        m = pre_res[:,1] < 0
        if len(m) > 0:
            pre_res[:,1][m] *= -1.
            pre_res[:,2][m] += np.pi
    
    return x, y, pre_res


def transit_search(idx):
    row = df.iloc[idx]
    try:
        fig = plt.figure(figsize=[8.27,11.69])

        # Load LC and process
        ax = plt.subplot(411)
        x, y, yerr = np.loadtxt(row.file).T
        x, y, yerr, smooth = preprocess_lc(x,y, yerr, ax=ax)

        # Prewhiten
        ax = plt.subplot(412)
        f,a = amplitude_spectrum(x, y, fmax=48)
        ax.plot(f, a, c='black', lw=0.7)
        ax.set_xlim(0, 48)

        x, y, res = deep_clean(x, y)
        ax.scatter(res[:,0], res[:,1], marker='v')
        ax.set_xlabel(r'Frequency [day$^{-1}$]')
        ax.set_ylabel('Amplitude [ppt]')

        # BLS 
        m = np.zeros(len(x), dtype=bool)
        period_grid = np.exp(np.linspace(np.log(1), np.log(300), 100000))
        bls_results = []
        periods = []
        t0s = []
        depths = []

        for i in range(1):
            bls = BoxLeastSquares(x[~m], y[~m] - smooth[~m])
            bls_power = bls.power(period_grid, 0.1, oversample=20)
            bls_results.append(bls_power)

            # Save the highest peak as the planet candidate
            index = np.argmax(bls_power.power)
            periods.append(bls_power.period[index])
            t0s.append(bls_power.transit_time[index])
            depths.append(bls_power.depth[index])

            # Mask the data points that are in transit for this candidate
            m |= bls.transit_mask(x, periods[-1], 0.5, t0s[-1])

        for i in range(len(bls_results)):
            # Plot the periodogram
            ax = plt.subplot(425)
            ax.axvline(np.log10(periods[i]), color="C1", lw=5, alpha=0.8)
            ax.plot(np.log10(bls_results[i].period), bls_results[i].power, "k", lw=0.7)
            ax.annotate(
                "period = {0:.4f} d".format(periods[i]),
                (0, 1),
                xycoords="axes fraction",
                xytext=(5, -5),
                textcoords="offset points",
                va="top",
                ha="left",
            )
            ax.set_ylabel("bls power")
            ax.set_yticks([])
            ax.set_xlim(np.log10(period_grid.min()), np.log10(period_grid.max()))
            if i < len(bls_results) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("log10(period)")

            # Plot the folded transit
            ax = plt.subplot(426)
            p = periods[i]
            x_fold = (x - t0s[i] + 0.5 * p) % p - 0.5 * p
            m = np.abs(x_fold) < 0.4
            ax.errorbar(x_fold[m], y[m] - smooth[m] ,fmt=".k",yerr=yerr[m], markersize=1, elinewidth=0.1, zorder=1)

            # Overplot the phase binned light curve
            bins = np.linspace(-0.41, 0.41, 32)
            denom, _ = np.histogram(x_fold, bins)
            num, _ = np.histogram(x_fold, bins, weights=y - smooth)
            denom[num == 0] = 1.0
            ax.plot(0.5 * (bins[1:] + bins[:-1]), num / denom, color="C1", zorder=50)

            ax.set_xlim(-0.4, 0.4)
            ax.set_xlabel("Time since transit")
            ax.set_ylabel('Flux [ppt]')

        # River plot
        ax = plt.subplot(427)
        ax = plot_river(x, y, periods[0], t0s[0], ax=ax)
        ax.set_xlabel('Phase')
        ax.set_ylabel('Cycle')
        ax.set_xlim(-0.2, 0.2)

        # Stats
        ax = plt.subplot(428)

        ax.text(0.1,0.8,f'KIC {row.kic}')
        ax.text(0.1,0.7,f'Teff: {row.Teffi}$\pm${row.e_Teffi} K')
        ax.text(0.1,0.6,f'Lum: {row.loglbol_g_median:.1f}$\pm${row.loglbol_sigm:.1f} $L/L_\odot$')
        ax.text(0.1,0.5,f'Mass: {row.new_mass:.1f}$\pm${row.new_mass_std:.1f} $M/M_\odot$')

        ax.text(0.1,0.3,f'Period: {periods[0]:.2f} d')
        ax.text(0.1,0.2,f't0: {t0s[0]:.2f} d')
        ax.text(0.1,0.1,f'Depth: {depths[0]:.2f} ppt')
        ax.axis('off')

        plt.savefig(f'res/plots/{row.kic}.png', dpi=300, bbox_inches='tight')

        plt.clf()
        plt.close('all')

        np.savetxt(f'res/prewhitened/{row.kic}.txt', list(zip(x, y, yerr)))
        np.savetxt(f'res/frequencies/{row.kic}.txt', res)

        return
    except:
        plt.clf()
        plt.close('all')
        print(f"I failed on star {row.kic}")
        return row.kic
