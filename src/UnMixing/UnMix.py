import numpy as np
import pymc as pm
import arviz as az
import multiprocessing 
import sys
import warnings
# Suppress FutureWarning related to pandas Series, will be changed to polars in time {todo} change to polars...
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append('/Users/alexander_crane/Desktop/Research/UnMixing/')
from src.UnMixing import MixResults, Distribution_make

# multiprocessing.set_start_method('fork')

# dist 1 needs to be the targets and dist 2 needs to be the MLE of the inerts
def UnMix(dist1, dist2):
    print("UnMixing...")
    if dist1.type and dist2.type == "Truncated_Normal":
        with pm.Model() as model:
            w = pm.Dirichlet("w", a=np.array([10, 10]))
            mu1 = pm.Normal("mu1", mu=dist1.mu, sigma=dist1.sigma)
            sig1 = pm.HalfNormal("sig1", sigma=dist1.sigma)

            dist1_temp = pm.TruncatedNormal.dist(mu=mu1,
                                                 sigma=sig1,
                                                 lower=dist1.lower_cutoff,
                                                 upper=dist1.upper_cutoff)
            
            
            dist2_temp = pm.TruncatedNormal.dist(mu=dist2.mle['MLE'].iloc[0],
                                                 sigma=dist2.mle['MLE'].iloc[1],
                                                 lower=dist2.lower_cutoff,
                                                 upper=dist2.upper_cutoff)
            like = pm.Mixture(name="like",
                              w=w,
                              comp_dists=[dist1_temp,dist2_temp],
                              observed=dist1.values)
            print("Verbose Model Debug:\n" + str(model.debug()))
        with model:
            idata=pm.sample(draws=1000, tune=1000, target_accept=0.95)
        #pm.model_to_graphviz(model)
        #az.plot_posterior(idata) # gives HDI which is a log scale distribution
        #az.plot_forest(idata, var_names= ['mu','sigma'])
        mu_temp = idata.posterior["mu1"].mean(("chain", "draw"))
        sig_temp = idata.posterior["sig1"].mean(("chain", "draw"))
        w_temp = idata.posterior["w"].mean(("chain", "draw"))

        ### model checking
        # 1. convergance diagnostics --> did MCMC work
        # 2. goodness of fit --> does it corespond to what the data suggests
        # az.plot_trace(idata, var_names = ['mu', 'sigma']m backend_kwargs=dict(constrained_layout=True));
        # az.plot_energy(idata)
        # az. summary(idata) --> get a summary table of r_hat (Potential Scale reduction, should be all close to 1)
        print("Generating results...")
        res = MixResults.MixResults(w = w_temp, mu = mu_temp, sigma= sig_temp, target_vals = dist1.values, 
                                    target_mle = dist1.mle, inert_vals = dist2.values, inert_mle = dist2.mle)

    return res

#### Notes on pymc Mixtures
#   Main challenge in fitting Bayesian models is calculating the joint posterior distribution across the model
#   MCMC: Markov chain Monte Carlo simulates a Markov chain for which some function of interest is the unique, invariant, stationary distribution
#       - Markov chain: stochastic process (indexed set of random variables) that is "Markovian" (probability of some variable ,its value in the current time step is only dependent on the        previous step)
#            - Pr(Xt+1 = xt+1|Xt = xt, Xt-1 = xt-1,...,X0=x0) = Pr(Xt+1 = xt+1| Xt = xt) --> history does not matter, think monopoly lol, past doesn't matter
#       - Need Reversible Markov chains for this to work --> constructed to satisfy the detailed balance equation:
#           - pi(x)Pr(y|x) = pi(y)Pr(x|y) in which ~pi~ is the limiting distribution of the chain
#       Metropolis sampling: used to determine if you accept or reject --> very easy to impliment and general --> not used cause it is meh
#
#       Hamiltonian Monte Carlo: kinda of takes into account your model using the gradient to simulate a physical analogy of value that follows the gradient like a rock going down a hill
#               - does a better job at sticking to the data
#   
#       NUTS --> No-Uturn_sampler --> special Hamiltonian Monte Carlo: state of the art sampler
#               - best one by far, just hit believe button
#               - autotunes, pretty good 
#
# Divergences after tuning 
#   - divergence is when Hamiltonian Monte Carlo does a poor job of moving along, 5 out of 1000 is fine
#   - Setting higher target acceptance --> more baby steps it has to take, fits better but takes longer to run, 0.95 is pretty good
#
# BFMI --> Bayesian Fraction of Missing Information
#   - estimates how hard it is to sample level sets of the posterior at each iteration. It says how well momentum resampling matches the marginal energy distribution
#   - want values close to 1, although there is no rule of thumb for what is a good cutoff
#   - plot energy with az.plot energy and check for a goofy output (i.e. chains not being aroung 1, want them to overlap, DO NOT WANT: Bimodal or delta functions)
#
# Potential Scale Reduction
#   - essentially an analysis of varaiance of the trace (idata) compares variance within chain to between chains
#   - Compares chains to make sure they are similar --> want close to 1
#
# Goodness of Fit
#   - compare output of model to the data that was used to fit the model
#   - Bayes gives an automatic way of doing this
#       - sample_posterior_predictive function draws posterior predictive samples from all the observed variables in the model
#       - EX: ppc = pm.sample_posterior_predictive(trace, extend_inferencedata=True)
#       - az.plot_ppc(idata, kind = 'cumulative') --> blue is draws from the model, black line should be within the blue lines