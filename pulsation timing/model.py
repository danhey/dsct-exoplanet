import pymc3 as pm
import aesara_theano_fallback.tensor as tt

import pymc3_ext as pmx
import exoplanet as xo
import numpy as np


def planet_model(time, flux, freq, amps, period_t, period_sd, eccen_t):
    """A custom PyMC3 model for planetary transits.
    
    Args:
        time (array): Time values
        flux (array): Flux values corresponding to `time`
        freq (array): Frequencies in the pulsation model
        amps (array): Amplitudes in the pulsation model
        period_t (float): Orbital period from the transit
        period_sd (float): Standard deviation of the orbital period from the transit
        eccen_t (float): Eccentricity
    
    Returns:
        PyMC3 Model: A planet pulsation timing model.
    """
    if np.isnan(period_sd):
        period_sd = 1e-5
        
    with pm.Model() as model:
        period = pm.Normal("period", mu=period_t, sd=period_sd, testval=period_t)
        phi = xo.distributions.Angle("phi")
        logs_lc = pm.Normal("logsigma", mu=np.log(np.std(flux)), sd=10, testval=0.0)
        
#         asini = pm.Bound(pm.Flat, lower=0)("asini", shape=len(freq), testval=1e-5 + np.zeros(len(freq)))
        asini = pm.Bound(pm.Flat, lower=0)("asini", testval=1e-5)
        mean = pm.Normal("mean", mu=0.0, sd=10.0, testval=0.00)

        # Mean anom
        M = 2.0 * np.pi * time / period - phi

        if eccen_t == 0:
            eccen = 0
            omega = 0
            psi = -tt.sin(M)
        else:
            eccen = pm.Uniform("eccen", lower=1e-3, upper=0.99, testval=eccen_t)
            omega = xo.distributions.Angle("omega", testval=0.0)  # True anom
            kepler_op = xo.theano_ops.kepler.KeplerOp()
            sinf, cosf = kepler_op(M, eccen + np.zeros(len(time)))

            factor = 1.0 - tt.square(eccen)
            factor /= 1.0 + eccen * cosf
            psi = -factor * (sinf * tt.cos(omega) + cosf * tt.sin(omega))

#         tau = pm.Deterministic("tau", psi * asini)#[:, None])
        tau = psi * asini
#         tau = pm.Deterministic("tau", psi * asini[:, None])
        lognu = pm.Normal("lognu", mu=np.log(freq), sd=0.1, shape=len(freq))
        nu = pm.Deterministic("nu", tt.exp(lognu))
        factor = 2.0 * np.pi * nu

        arg = (factor * (1))[None, :] * time[:, None] - (factor * asini / 86400)[
            None, :
        ] * psi[:, None]

        phase = xo.distributions.Angle("phase", shape=len(freq))
        logamp = pm.Normal(
            "logamp", mu=np.log(amps), sd=0.01, shape=len(freq), testval=np.log(amps)
        )
#         lc_model = pm.Deterministic('lc',
#             tt.sum(tt.exp(logamp)[None, :] * tt.sin(arg - phase[None, :]), axis=1)
#             + mean
#         )
        lc_model = tt.sum(tt.exp(logamp)[None, :] * tt.sin(arg - phase[None, :]), axis=1) + mean

        # We pass this into our likelihood
        pm.Normal("obs", mu=lc_model, sd=tt.exp(logs_lc), observed=flux)

    return model