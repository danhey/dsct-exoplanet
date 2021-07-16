import glob
import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import exoplanet as xo
import pymc3 as pm
import aesara_theano_fallback.tensor as tt
import pickle # python3
import tqdm
import pymc3_ext as pmx
from celerite2.theano import terms, GaussianProcess
import arviz as az


def build_model(mask=None, start=None):
    if mask is None:
        mask = np.ones(len(x), dtype=bool)
    with pm.Model() as model:

        # Parameters for the stellar properties
        mean = pm.Normal("mean", mu=0.0, sd=10.0)
        u_star = xo.QuadLimbDark("u_star")

        # Stellar parameters from Huang et al (2018)
#         M_star_huang = row.new_mass, row.new_mass_std
#         R_star_huang = row.new_radius, row.new_r_std
#         BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=3)
#         m_star = BoundedNormal(
#             "m_star", mu=M_star_huang[0], sd=M_star_huang[1]
#         )
#         r_star = BoundedNormal(
#             "r_star", mu=R_star_huang[0], sd=R_star_huang[1]
#         )

        # Orbital parameters for the planets
        t0 = pm.Normal("t0", mu=bls_t0, sd=1)
        log_period = pm.Normal("log_period", mu=np.log(bls_period), sd=1)
        log_ror = pm.Normal(
            "log_ror", mu=0.5 * np.log(bls_depth * 1e-3), sigma=10.0
        )
        ror = pm.Deterministic("ror", tt.exp(log_ror))
        
        period = pm.Deterministic("period", tt.exp(log_period))
        b = xo.distributions.ImpactParameter("b", ror=ror)
        
        log_dur = pm.Normal("log_dur", mu=np.log(0.1), sigma=10.0)
        dur = pm.Deterministic("dur", tt.exp(log_dur))
        
        # Transit jitter & GP parameters
        log_sigma_lc = pm.Normal(
            "log_sigma_lc", mu=np.log(np.std(y[mask])), sd=10
        )
        log_rho_gp = pm.Normal("log_rho_gp", mu=0, sd=10)
        log_sigma_gp = pm.Normal(
            "log_sigma_gp", mu=np.log(np.std(y[mask])), sd=10
        )

        
        # Orbit model
        orbit = xo.orbits.KeplerianOrbit(
            period=period,
            t0=t0,
            b=b,
            duration=dur
        )
        

        # Compute the model light curve
        light_curves = pm.Deterministic(
            "light_curves",
            xo.LimbDarkLightCurve(u_star).get_light_curve(
                orbit=orbit, r=ror, t=x[mask], 
                texp=texp
            )
            * 1e3,
        )
        light_curve = tt.sum(light_curves, axis=-1) + mean
        resid = y[mask] - light_curve
        
        # GP model for the light curve
        kernel = terms.SHOTerm(
            sigma=tt.exp(log_sigma_gp),
            rho=tt.exp(log_rho_gp),
            Q=1 / np.sqrt(2),
        )
        gp = GaussianProcess(kernel, t=x[mask], yerr=tt.exp(log_sigma_lc))
        gp.marginal("gp", observed=resid)
        pm.Deterministic("gp_pred", gp.predict(resid))

        # Fit for the maximum a posteriori parameters, I've found that I can get
        # a better solution by trying different combinations of parameters in turn
        if start is None:
            start = model.test_point
        map_soln = pmx.optimize(
            start=start, vars=[log_sigma_lc, log_sigma_gp, log_rho_gp]
        )
        map_soln = pmx.optimize(start=map_soln, vars=[dur, log_ror])
        map_soln = pmx.optimize(start=map_soln, vars=[b])
        map_soln = pmx.optimize(start=map_soln, vars=[log_period, t0])
        map_soln = pmx.optimize(start=map_soln, vars=[u_star])
        map_soln = pmx.optimize(start=map_soln, vars=[dur, log_ror])
        map_soln = pmx.optimize(start=map_soln, vars=[b])
#         map_soln = pmx.optimize(start=map_soln, vars=[ecs])
        map_soln = pmx.optimize(start=map_soln, vars=[dur])
        map_soln = pmx.optimize(start=map_soln, vars=[mean])
        map_soln = pmx.optimize(
            start=map_soln, vars=[log_sigma_lc, log_sigma_gp, log_rho_gp]
        )
        map_soln = pmx.optimize(start=map_soln)

    return model, map_soln


def plot_light_curve(soln, mask=None):
    if mask is None:
        mask = np.ones(len(x), dtype=bool)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    ax = axes[0]
    ax.plot(x[mask], y[mask], "k", label="data")
    gp_mod = soln["gp_pred"] + soln["mean"]
    ax.plot(x[mask], gp_mod, color="C2", label="gp model")
    ax.legend(fontsize=10)
    ax.set_ylabel("relative flux [ppt]")

    ax = axes[1]
    ax.plot(x[mask], y[mask] - gp_mod, "k", label="de-trended data")
    for i, l in enumerate("a"):
        mod = soln["light_curves"][:, i]
        ax.plot(x[mask], mod, label="planet model")
    ax.legend(fontsize=10, loc=3)
    ax.set_ylabel("de-trended flux [ppt]")

    ax = axes[2]
    mod = gp_mod + np.sum(soln["light_curves"], axis=-1)
    ax.plot(x[mask], y[mask] - mod, "k")
    ax.axhline(0, color="#aaaaaa", lw=1)
    ax.set_ylabel("residuals [ppt]")
    ax.set_xlim(x[mask].min(), x[mask].max())
    ax.set_xlabel("time [days]")

    return fig

# plt.style.use('science')
print("I begin")
df = pd.read_csv("../data/catalogues/rev_AF_stars.csv", dtype={'kic': str})
candidates = pd.read_csv("candidates.csv", dtype={'kic': str}, delimiter='\t')

kicid = [
# '9471419',
# '11666429',
'9895543',
# # '9875566',
# # '8456151',
# '8249829',
# '8057661',
# # '7975162',
# '7767699',
# '5724523',
# '4380834',
# '10071056',
# '12116239',
]
candidates = candidates[candidates.kic.isin(kicid)]

df = df.merge(candidates, left_on='kic', right_on='kic')

for index, row in tqdm.tqdm(df[:].iterrows(), total=len(df[:])):
    np.random.seed(42)
    f = glob.glob(f"../prewhitening/results/spline and BIC/prewhitened/*{row.kic}*")[0]
    time, flux, flux_err = np.loadtxt(f, unpack=True)
    lc = lk.LightCurve(time, flux + 1., flux_err)
    lc = lc.flatten(window_length=201, break_tolerance=11).remove_outliers(sigma_upper=5, sigma_lower=5)
    time, flux, flux_err = lc.time.value, lc.flux.value, lc.flux_err.value
    flux_err /= 1e3
    flux -= 1.

    m = np.ones_like(time, dtype=bool)
    ref_time = 0.5 * (np.min(time) + np.max(time))
    x = np.ascontiguousarray(time[m] - ref_time, dtype=np.float64)
    y = np.ascontiguousarray(flux[m], dtype=np.float64)
    yerr = np.ascontiguousarray(flux_err[m], dtype=np.float64)
    #
    texp = np.min(np.diff(x))
    
    bls_period = row.period
    bls_t0 = row.t0
    bls_depth = row.bls_depth
    
    
    model0, map_soln0 = build_model()

    mod = (
        map_soln0["gp_pred"]
        + map_soln0["mean"]
        + np.sum(map_soln0["light_curves"], axis=-1)
    )
    resid = y - mod
    rms = np.sqrt(np.median(resid ** 2))
    mask = np.abs(resid) < 5 * rms

    model, map_soln = build_model(mask, map_soln0)

    
    # Plot initial LC:
    plt.figure(figsize=(8, 4))


    # Compute the GP prediction
    gp_mod = map_soln["gp_pred"] + map_soln["mean"]

    # Get the posterior median orbital parameters
    p = map_soln["period"]
    t0 = map_soln["t0"]

    # Plot the folded data
    x_fold = (x[mask] - t0 + 0.5 * p) % p - 0.5 * p
    # plt.plot(x_fold, y[mask]- gp_mod, 
    #          ".k", label="data", zorder=-1000)
    plt.scatter(x_fold, y[mask]- gp_mod, c=x[mask], s=3, alpha=0.5,
             label="data", zorder=-1000)
    plt.colorbar(label="time [days]")
    # Plot the folded model
    inds = np.argsort(x_fold)
    inds = inds[np.abs(x_fold)[inds] < 0.3]
    pred = map_soln["light_curves"][inds, 0]

    plt.plot(x_fold[inds], pred, color="black", linewidth=2, label="model")

    # Annotate the plot with the planet's period
    txt = "period = {0:.5f} d".format(
        map_soln["period"], #map_soln["period"]
    )
    plt.annotate(
        txt,
        (0, 0),
        xycoords="axes fraction",
        xytext=(5, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=12,
    )

    plt.legend(fontsize=10, loc=4)
    plt.xlim(-0.5 * p, 0.5 * p)
    plt.xlabel("time since transit [days]")
    plt.ylabel("de-trended flux")
    _ = plt.xlim(-0.2, 0.2)

    plt.savefig(f'plots/{row.kic}_initial.png', dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close('all')
    
    np.random.seed(1)
    print(map_soln)
    with model:
        trace = pmx.sample(
            tune=1000,
            draws=1000,
            start=map_soln,
            cores=2,
            chains=2,
            initial_accept=0.8,
            target_accept=0.95,
            return_inferencedata=False,
        )

    with open(f'traces/{row.kic}.pkl', 'wb') as buff:
        pickle.dump({'model': model, 'trace': trace, 'time': x, 'flux': y, 'mask': mask}, buff)

    with model:
        trace = az.from_pymc3(trace)

    plt.figure(figsize=[7,7])
    flat_samps = trace.posterior.stack(sample=("chain", "draw"))

    # Compute the GP prediction
    gp_mod = np.median(
        flat_samps["gp_pred"].values + flat_samps["mean"].values[None, :], axis=-1
    )

    # Get the posterior median orbital parameters
    p = np.median(flat_samps["period"])
    t0 = np.median(flat_samps["t0"])

    # Plot the folded data
    x_fold = (x[mask] - t0 + 0.5 * p) % p - 0.5 * p
    plt.plot(x_fold, y[mask] - gp_mod, ".k", label="data", zorder=-1000)

    # Overplot the phase binned light curve
    bins = np.linspace(-0.41, 0.41, 50)
    denom, _ = np.histogram(x_fold, bins)
    num, _ = np.histogram(x_fold, bins, weights=y[mask])
    denom[num == 0] = 1.0
    plt.plot(
        0.5 * (bins[1:] + bins[:-1]), num / denom, "o", color="C2", label="binned"
    )

    # Plot the folded model
    inds = np.argsort(x_fold)
    inds = inds[np.abs(x_fold)[inds] < 0.3]
    pred = np.percentile(
        flat_samps["light_curves"][inds, 0], [16, 50, 84], axis=-1
    )
    plt.plot(x_fold[inds], pred[1], color="C1", label="model")
    art = plt.fill_between(
        x_fold[inds], pred[0], pred[2], color="C1", alpha=0.5, zorder=1000
    )
    art.set_edgecolor("none")

    # Annotate the plot with the planet's period
    txt = "period = {0:.5f} +/- {1:.5f} d".format(
        np.mean(flat_samps["period"].values), np.std(flat_samps["period"].values)
    )
    plt.annotate(
        txt,
        (0, 0),
        xycoords="axes fraction",
        xytext=(5, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=12,
    )

    plt.legend(fontsize=10, loc=4)
    plt.xlim(-0.5 * p, 0.5 * p)
    plt.xlabel("time since transit [days]")
    plt.ylabel("de-trended flux")
    _ = plt.xlim(-0.15, 0.15)


    plt.savefig(f'plots/{row.kic}.png', dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close('all')