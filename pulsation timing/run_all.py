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

koi_dsct = pd.read_csv('data/koi_dsct.csv')

for index, row in koi_dsct.iterrows():

    koi_period = row.koi_period
    koi_period_err = row.koi_period_err1
    koi_eccen = row.koi_eccen

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
        
    with open(f'traces/{row.kepoi_name}.pkl', 'wb') as buff:
        pickle.dump({'model': model, 'trace': trace}, buff)