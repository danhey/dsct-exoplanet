import exoplanet as xo
import pymc3 as pm
import aesara_theano_fallback.tensor as tt

import pymc3_ext as pmx
from celerite2.theano import terms, GaussianProcess


def build_model(mask=None, start=None, optimize=False):
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
        if optimize:
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
        else:
            map_soln = start

    return model, map_soln