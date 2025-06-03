import emcee
import corner
import numpy as np
from multiprocessing import Pool
from astropy.table import Table

parentDir  ='/Users/macbookpro/projects/criticalMassLRD'
dataAkins24=Table.read(parentDir+'/data'+'/akinsDataMCMC.tex',format='latex')

logMAkins       = dataAkins24['logM'      ][:]
logMAkins_error = dataAkins24['logM_error'][:]
logRAkins       = dataAkins24['logR'      ][:]
logRAkins_error = dataAkins24['logR_error'][:]
log1pz          = np.log10(dataAkins24['redshift'  ][:]+1)

N= len(logMAkins)

def log_prior(theta):
    a, b, c, lnsigma_int = theta[:4]
    if not (-5 < a < 5 and -5 < b < 5 and -10 < c < 10 and -10 < lnsigma_int < 1):
        return -np.inf
    return 0.0  # flat prior

def log_likelihood(theta, logM_obs, logM_err, logR_obs, logR_err, log1pz):
    a, b, c, lnsigma_int = theta[:4]
    logM_true = theta[4:4+N]
    logR_true = theta[4+N:]
    sigma_int = np.exp(lnsigma_int)

    # Likelihood of observed M_star given latent logM_true
    logL_mass = -0.5 * np.sum((logM_obs - logM_true)**2 / logM_err**2 + np.log(2*np.pi*logM_err**2))

    # Likelihood of observed R given latent logR_true
    logL_radius = -0.5 * np.sum((logR_obs - logR_true)**2 / logR_err**2 + np.log(2*np.pi*logR_err**2))

    # Latent radius model
    model_logR = a * logM_true + b * log1pz + c
    logL_intrinsic = -0.5 * np.sum((logR_true - model_logR)**2 / sigma_int**2 + np.log(2*np.pi*sigma_int**2))

    return logL_mass + logL_radius + logL_intrinsic

def log_posterior(theta, logM_obs, logM_err, logR_obs, logR_err, log1pz):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, logM_obs, logM_err, logR_obs, logR_err, log1pz)
    return lp + ll

ndim = 4 + 2 * N
nwalkers = 2 * ndim

# Initial guess
a0, b0, c0, sigma0 = 0.3, -1.0, 0.0, 0.1
logM_init = logMAkins.copy()
logR_init = logRAkins.copy()
p0 = np.hstack([a0, b0, c0, np.log(sigma0), logM_init, logR_init])
pos = p0 + 1e-3 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior,
                                args=(logMAkins, logMAkins_error, logRAkins, logRAkins_error, log1pz))

    
ncores = multiprocessing.cpu_count()

with Pool(processes=ncores) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=args, pool=pool)
    sampler.run_mcmc(pos, 5000, progress=True)

# Save the full chain and log probabilities

np.save("mcmc_chain.npy", sampler.get_chain())              # shape (nsteps, nwalkers, ndim)
np.save("mcmc_log_prob.npy", sampler.get_log_prob())        # shape (nsteps, nwalkers)

# Optional: flatten and save samples after burn-in
samples = sampler.get_chain(discard=1000, thin=10, flat=True)
np.save("mcmc_samples_flat.npy", samples)

# Save parameter labels
labels = ["a", "b", "c", "ln σ_int"]
np.save("mcmc_labels.npy", labels)


samples = sampler.get_chain(discard=1000, thin=10, flat=True)
model_samples = samples[:, :4]  # a, b, c, lnsigma_int

labels = ["a", "b", "c", "ln σ_int"]
corner.corner(model_samples, labels=labels)


