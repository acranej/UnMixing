import pandas as pd
import pymc as pm
import arviz as az
import numpy as np
import os
from scipy.stats import norm, expon, gamma, lognorm
from math import log, exp
from pytensor.tensor import TensorVariable
import matplotlib.pyplot as plt
import pickle
import sys

def dist_expo(
    lam_expo: TensorVariable,
    shift_expo: TensorVariable,
    size_expo: TensorVariable,
) -> TensorVariable:
    return pm.Exponential.dist(lam_expo, size = size_expo) + shift_expo

def dist_gam(
    alpha_expo: TensorVariable,
    beta_expo: TensorVariable,
    shift_gam: TensorVariable,
    size_gam: TensorVariable,
) -> TensorVariable:
    return pm.Gamma.dist(alpha = alpha_expo, beta=beta_expo, size=size_gam) + shift_gam

def dist_lognormal(
    mu_ln: TensorVariable,
    sigma_ln: TensorVariable,
    shift_ln: TensorVariable,
    size_ln: TensorVariable
) -> TensorVariable:
    return pm.LogNormal.dist(mu=mu_ln, sigma=sigma_ln, size=size_ln) + shift_ln

tumors = pd.read_pickle('/Users/alexander_crane/Desktop/Research/Inerts/Jeff/constrained_tumors_test_low_cutoff.pkl')
tumors = tumors[tumors != 1]
tumors = tumors.to_frame().reset_index()
tumors_KT = tumors[tumors['Time'] == 2] #array([ 2.,  4.,  8., 12., 18., 24., 32.])
KT_df = tumors_KT[tumors_KT['Genotype'] == 'KT']
unique_mice = KT_df['Mouse'].unique()
print(unique_mice)
#unique_mice = ['3570']
def main():
    for mouse in unique_mice:
        plot_num = 1
        KT_temp_df = KT_df[(KT_df.Mouse == mouse)]
        KT_temp_df.rename(columns={0: 'Cells'}, inplace = True)
        temp_list = KT_temp_df.Cells.tolist()
        temp_list_ = [log(x) for x in temp_list]
        dat_full = np.array(temp_list_)
        os.chdir('/Users/alexander_crane/Desktop/Research/KT_Mix_Analysis/Compare_Fits/LessRestrictedBounds/Week2')
        os.makedirs(mouse, exist_ok=True)
        os.chdir(mouse)

        print('Gauss and shifted gamma...')

        with pm.Model() as modelgam:
            w = pm.Dirichlet('w', a=np.array([1, 1]))
            shift = pm.Normal('shift', 
                                mu = (((np.mean(dat_full) + (2.6*np.std(dat_full))) + (np.mean(dat_full))) / 2), # mid point of upper and lower bounds
                                sigma = (((np.mean(dat_full) + (2.6*np.std(dat_full))) - (np.mean(dat_full))) / 6))
            alpha = pm.HalfNormal('alpha', sigma= 2.5)
            beta = pm.HalfNormal('beta', sigma = 2.5)
            shiftedgamma = pm.CustomDist.dist(alpha, beta, shift, dist=dist_gam)
            mu1 = pm.Normal("mu1", mu=np.mean(dat_full), sigma=np.std(dat_full))
            sig1 = pm.HalfNormal('sig1', sigma=np.std(dat_full))
            gaussian = pm.Normal.dist(mu=mu1, sigma=sig1)
            like_gaus_shiftedgamma = pm.Mixture(name='Gaus_ShiftedGam',
                                            w=w,
                                            comp_dists=[gaussian, shiftedgamma],
                                            observed=dat_full)
        with modelgam:
            gaus_shiftedgam_trace = pm.sample_smc(random_seed=3) #100 looked nice ###################################
            pm.compute_log_likelihood(gaus_shiftedgam_trace)

        print('Gaussian + shifted expo')
        with pm.Model() as modelexpo:
            w = pm.Dirichlet('w', a=np.array([1, 1]))
            lambda_ = pm.HalfNormal('lambda', sigma = 2.5) 
            shift = pm.Normal('shift', 
                                mu = (((np.mean(dat_full) + (2.6*np.std(dat_full))) + (np.mean(dat_full))) / 2), # mid point of upper and lower bounds
                                sigma = (((np.mean(dat_full) + (2.6*np.std(dat_full))) - (np.mean(dat_full))) / 6))
            shiftedexpo = pm.CustomDist.dist(lambda_, shift, dist=dist_expo)
            mu1 = pm.Normal("mu1", mu=np.mean(dat_full), sigma=np.std(dat_full))
            sig1 = pm.HalfNormal('sig1', sigma=np.std(dat_full))
            gaussian = pm.Normal.dist(mu=mu1, sigma=sig1)
            like_gaus_shiftedexpo = pm.Mixture(name='Gaus_ShiftedExpo',
                                            w=w,
                                            comp_dists=[gaussian, shiftedexpo],
                                            observed=dat_full)
        with modelexpo:
            gaus_shiftedexpo_trace = pm.sample_smc(random_seed=3) #100 looked nice############################################
            pm.compute_log_likelihood(gaus_shiftedexpo_trace)
        
        print('Gaussian + log normal')
        with pm.Model() as modellognorm:
            w= pm.Dirichlet('w', a=np.array([1, 1]))
            mu_ln = pm.HalfNormal('mu_ln', sigma = 2.5)
            sig_ln = pm.HalfNormal('sig_ln', sigma = 2.5)
            shift = pm.Normal('shift', 
                                mu = (((np.mean(dat_full) + (2.6*np.std(dat_full))) + (np.mean(dat_full))) / 2), # mid point of upper and lower bounds
                                sigma = (((np.mean(dat_full) + (2.6*np.std(dat_full))) - (np.mean(dat_full))) / 6))
            
            log_norm = pm.CustomDist.dist(mu_ln, sig_ln, shift, dist=dist_lognormal)
            mu1 = pm.Normal("mu1", mu=np.mean(dat_full), sigma=np.std(dat_full))
            sig1 = pm.HalfNormal('sig1', sigma=np.std(dat_full))
            gaussian = pm.Normal.dist(mu=mu1, sigma=sig1)
            like_gaus_ln = pm.Mixture(name='Gaus_LogNormal',
                                            w=w,
                                            comp_dists=[gaussian, log_norm],
                                            observed=dat_full)
        with modellognorm:
            gaus_lognorm_trace = pm.sample_smc(random_seed=3) #100 looked nice############################################
            pm.compute_log_likelihood(gaus_lognorm_trace)

        
        #### extract values for plotting ###############
        #gaus + shifted gamma
        w_temp_gam = gaus_shiftedgam_trace.posterior["w"].mean(("chain","draw"))
        alpha_temp_gam = gaus_shiftedgam_trace.posterior["alpha"].mean(("chain","draw"))
        beta_temp_gam = gaus_shiftedgam_trace.posterior["beta"].mean(("chain","draw"))
        shift_temp_gam = gaus_shiftedgam_trace.posterior["shift"].mean(("chain","draw"))
        mu_temp_gam = gaus_shiftedgam_trace.posterior["mu1"].mean(("chain","draw"))
        sig_temp_gam = gaus_shiftedgam_trace.posterior["sig1"].mean(("chain","draw"))

        #gaus + shifted expo
        w_temp_expo = gaus_shiftedexpo_trace.posterior["w"].mean(("chain","draw"))
        lambda_temp_expo = gaus_shiftedexpo_trace.posterior["lambda"].mean(("chain","draw"))
        shift_temp_expo = gaus_shiftedexpo_trace.posterior["shift"].mean(("chain","draw"))
        mu_temp_expo = gaus_shiftedexpo_trace.posterior["mu1"].mean(("chain","draw"))
        sig_temp_expo = gaus_shiftedexpo_trace.posterior["sig1"].mean(("chain","draw"))

        #gaus + log normal
        w_temp_ln = gaus_lognorm_trace.posterior["w"].mean(("chain","draw"))
        mu_ln_temp_ln = gaus_lognorm_trace.posterior["mu_ln"].mean(("chain","draw"))
        sig_ln_temp_ln = gaus_lognorm_trace.posterior["sig_ln"].mean(("chain","draw"))
        shift_temp_ln = gaus_lognorm_trace.posterior["shift"].mean(("chain","draw"))
        mu_temp_ln = gaus_lognorm_trace.posterior["mu1"].mean(("chain","draw"))
        sig_temp_ln = gaus_lognorm_trace.posterior["sig1"].mean(("chain","draw"))
        
        ########################
        ## values for plotting#####
        x = np.linspace(1,15,1000)
        line_width = 0.5
        ################
        print('Plotting unmix...')

        # gaus + shifted gamma
        gaussian_density_gam = w_temp_gam.values[0]*norm.pdf(x, mu_temp_gam, sig_temp_gam)
        shifted_gamma_density = (1-w_temp_gam.values[0]) * gamma.pdf(x, a=alpha_temp_gam, loc=shift_temp_gam, scale= 1/beta_temp_gam)
        combined_density_gam = gaussian_density_gam + shifted_gamma_density
        plt.subplot(4,2,plot_num)
        plt.hist(dat_full, density=True, bins = 40, color = "black")
        plt.plot(x, gaussian_density_gam, color = 'blue', label = f'Gaussian: {w_temp_gam.values[0]:.2f}', linewidth= line_width)
        plt.plot(x, shifted_gamma_density, color='green', label=f'Shifted Gamma: {(1-w_temp_gam.values[0]):.2f}',linewidth= line_width)
        plt.title('Gaussian + Shifted_Gamma')
        plt.legend(fontsize='small')
        plot_num +=1

        plt.subplot(4,2,plot_num)
        plt.hist(dat_full, density=True, bins = 40, color = "black")
        plt.plot(x, combined_density_gam, color = 'red', label = 'Combined Density', linewidth=line_width)
        plt.legend(fontsize='small')
        plot_num +=1

        # gaus + shifted expo
        gaussian_density_expo = w_temp_expo.values[0]*norm.pdf(x, mu_temp_expo, sig_temp_expo)
        shifted_expo_density = (1-w_temp_expo.values[0]) * expon.pdf(x, loc=shift_temp_expo, scale= 1/lambda_temp_expo)
        combined_density_expo = gaussian_density_expo + shifted_expo_density
        plt.subplot(4,2,plot_num)
        plt.hist(dat_full, density=True, bins = 40, color = "black")
        plt.plot(x, gaussian_density_expo, color = 'blue', label = f'Gaussian: {w_temp_expo.values[0]:.2f}', linewidth=line_width)
        plt.plot(x, shifted_expo_density, color='green', label= f'Shifted Expo: {(1-w_temp_expo.values[0]):.2f}', linewidth=line_width)
        plt.title('Gaussian + Shifted_Expo')
        plt.legend(fontsize='small')
        plot_num +=1

        plt.subplot(4,2,plot_num)
        plt.hist(dat_full, density=True, bins = 40, color = "black")
        plt.plot(x, combined_density_expo, color = 'red', label = 'Combined Density', linewidth=line_width)
        plt.legend(fontsize='small')
        plot_num +=1

        # gaus + lognormal
        gaussian_density_ln = w_temp_ln.values[0] * norm.pdf(x, mu_temp_ln, sig_temp_ln)
        lognorm_density = (1-w_temp_ln.values[0]) * lognorm.pdf(x, s= sig_ln_temp_ln, loc= shift_temp_ln, scale =  np.exp(mu_ln_temp_ln))
        combined_density_ln = gaussian_density_ln + lognorm_density
        plt.subplot(4,2,plot_num)
        plt.hist(dat_full, density=True, bins = 40, color = "black")
        plt.plot(x, gaussian_density_ln, color = 'blue', label = f'Gaussian: {w_temp_ln.values[0]:.2f}', linewidth=line_width)
        plt.plot(x, lognorm_density, color='green', label = f'LogNormal: {(1-w_temp_ln.values[0]):.2f}', linewidth= line_width)
        plt.title('Gaussian + LogNormal')
        plt.legend(fontsize = 'small')
        plot_num +=1

        plt.subplot(4,2,plot_num)
        plt.hist(dat_full, density=True, bins = 40, color = "black")
        plt.plot(x, combined_density_ln, color = 'red', label = 'Combined Density', linewidth=line_width)
        plt.legend(fontsize='small')
        plot_num +=1

        plt.suptitle(mouse)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
        plt.tight_layout()
        plt.savefig(mouse + '.pdf')
        plt.close()

        df_comp_loo = az.compare({'Gaus_Expo': gaus_shiftedexpo_trace, "Gaus_Gam": gaus_shiftedgam_trace, "Gaus_LogNorm": gaus_lognorm_trace})
        comp = az.plot_compare(df_comp_loo, insample_dev=False)
        plt.savefig(mouse +"_modelcomp.pdf", bbox_inches='tight', pad_inches=0.4)
        plt.close()
        df_comp_loo.to_csv(mouse +'_modelcomp.csv', sep = '\t')

        ax = az.plot_trace(gaus_shiftedgam_trace, compact=False, kind="rank_vlines")
        plt.tight_layout()
        plt.savefig(mouse + "gausShiftedGamma_rankplotTrace.pdf")
        plt.close()

        ax = az.plot_trace(gaus_shiftedexpo_trace, compact=False, kind="rank_vlines")
        plt.tight_layout()
        plt.savefig(mouse + "_gausShiftedExpo_rankplotTrace.pdf")
        plt.close() 

        ax = az.plot_trace(gaus_lognorm_trace, compact=False, kind="rank_vlines")
        plt.tight_layout()
        plt.savefig(mouse + "_gausLogNormal_rankplotTrace.pdf")
        plt.close()
        output_gaus_shifted_gam = az.summary(gaus_shiftedgam_trace, round_to =3)
        output_gaus_shifted_gam_file = 'output_gaus_shifted_gam.txt'
        with open(output_gaus_shifted_gam_file, 'w') as file:
            original_stdout = sys.stdout
            try:
                sys.stdout = file
                print(output_gaus_shifted_gam)
                print(az.loo(gaus_shiftedgam_trace))
            finally:
                sys.stdout = original_stdout

        output_gaus_shifted_expo = az.summary(gaus_shiftedexpo_trace, round_to =3)
        output_gaus_shifted_expo_file = 'output_gaus_shifted_expo.txt'
        with open(output_gaus_shifted_expo_file, 'w') as file:
            original_stdout = sys.stdout
            try:
                sys.stdout = file
                print(output_gaus_shifted_expo)
                print(az.loo(gaus_shiftedexpo_trace))
            finally:
                sys.stdout = original_stdout

        output_gaus_shifted_ln = az.summary(gaus_lognorm_trace, round_to =3)
        output_gaus_shifted_ln_file = 'output_gaus_lognormal.txt'
        with open(output_gaus_shifted_ln_file, 'w') as file:
            original_stdout = sys.stdout
            try:
                sys.stdout = file
                print(output_gaus_shifted_ln)
                print(az.loo(gaus_lognorm_trace))
            finally:
                sys.stdout = original_stdout

if __name__ == '__main__':
    main()
        

  
