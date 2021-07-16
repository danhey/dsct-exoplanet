import glob
import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
from utils import estimate_frequencies, amplitude_spectrum, high_pass, get_kepler_lc, preprocess_lc

import pymc3 as pm
import aesara_theano_fallback.tensor as tt

import pymc3_ext as pmx
import exoplanet as xo
from model import planet_model
import pickle # python3
import glob

cand = pd.DataFrame({
    'kepid': ['4380834', '5724523', '7767699', '8249829', '9895543'],
    'period': [0.889896, 0.545784, 1.128263, 1.012737, 1.218900],
    'e_period': [0.000023, 0.000010, 0.000037, 0.000101, 0.000025],
    'nfreqs': [
    2,
    3,
    3,
    1,
    2
]
})

for index, row in cand.iterrows():

    koi_period = row.period
    koi_period_err = row.e_period
    koi_eccen = 0

    time, flux = get_kepler_lc(row.kepid)
    time, flux = preprocess_lc(time, flux)
    flux *= 1e3

    peaks = estimate_frequencies(time, flux, max_peaks=row.nfreqs)
    plt.plot(*amplitude_spectrum(time, flux, fmax=48))
    for p in peaks:
        plt.axvline(p, c='red', lw=5, alpha=0.25)

    amps = np.array([amplitude_spectrum(time, flux, freq=[p])[1] for p in peaks]).squeeze()
    plt.savefig(f'traces/{row.kepid}.png')
    plt.clf()
    plt.close('all')
    
    
    model = planet_model(time, flux, peaks, amps, koi_period, koi_period_err, 0)

    with model:
        all_but = [v for v in model.vars if v.name not in ["logP_interval__", "asini_interval__"]]

        map_params = xo.optimize(start=None, vars=[model['mean']])
        map_params = xo.optimize(start=map_params, vars=[model['logsigma']])
        map_params = xo.optimize(start=map_params, vars=[model['phase'], model['logamp']])
        if not koi_eccen == 0:
            map_params = xo.optimize(start=map_params, vars=[model['eccen'], model['omega']])
        map_params = xo.optimize(start=map_params, vars=[model["phi"]])
        map_params = xo.optimize(start=map_params, vars=[model["lognu"]])
        map_params = xo.optimize(start=map_params)

        map_params = xo.optimize(start=map_params, vars=[model['asini']])
        map_params = xo.optimize(start=map_params)
        
        print(map_params)
        
        trace = pmx.sample(
            tune=1000,
            draws=1000,
            start=map_params,
            cores=2,
            chains=2,
            initial_accept=0.8,
            target_accept=0.95,
            return_inferencedata=False,
        )
        
    with open(f'traces/{row.kepid}.pkl', 'wb') as buff:
        pickle.dump({'model': model, 'trace': trace}, buff)