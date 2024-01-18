import scipy.stats as stats
import pandas as pd
import numpy as np
def g_z_mGal(z, a, rho, x, c=0):
    # c: center location
    G = 6.67e-11
    # 0.00001 m/s2
    gz = G*(4/3)*np.pi*(a**3)*rho*z/(((x-c)**2+z**2)**(3/2))*1000*100 
    return gz
def ModelError_BayesF(d_pos, d_fobs):
    '''
    Bayes factor for model error quantification. Assumption: Gaussian. 
    d_pos: samples of posterior data @1 station
    d_fobs: samples of observed data @1 station
    
    Return: Bayes factor
    '''
    
    fdobs_mean = d_fobs.mean()
    fdobs_std = d_fobs.std()
    
    f_obs_model = stats.norm.pdf(d_pos, loc=fdobs_mean, scale=fdobs_std)
    f_obs_dobs = stats.norm.pdf(d_fobs, loc=fdobs_mean, scale=fdobs_std)
    h1_indep = np.mean(f_obs_model, axis=0)
    h2_indep = np.mean(f_obs_dobs, axis=0)
    if h2_indep!=0:
        bayes_f = h1_indep/h2_indep
    else: 
        bayes_f = 10
    return bayes_f
def two_sphere_gravity_fwrd(theta_input, x1 = -15, x2 = 70):
    # run the forward modeling
    gravity_ = []

    for i in range(len(theta_input)):
        d_1 = g_z_mGal(theta_input['z1 (m)'][i], 
                       theta_input['a1 (m)'][i], 
                       theta_input['rho1 (g/cm3)'][i], x, c=x1)
        d_2 = g_z_mGal(theta_input['z2 (m)'][i], 
                       theta_input['a2 (m)'][i], 
                       theta_input['rho2 (g/cm3)'][i], x, c=x2)
        gravity_.append(d_1+d_2)
    return np.asarray(gravity_)