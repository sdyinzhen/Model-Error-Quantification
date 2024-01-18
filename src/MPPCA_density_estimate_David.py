from src.mppca_vDavid import *
from scipy.stats import multivariate_normal
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
class MPPCA_density_estimate_David():
    '''This version uses the mppca_vDavid to calculate MPPCA parameters'''
    def __init__(self, m_input, d_input, M, q, niter = 100):
        self.d_input = d_input
        self.m_input = m_input
        self.M = M
        self.q = q
        self.niter = niter
        train_data = np.c_[m_input, d_input]
        '''Train the MPPCA to estimate parameters'''
        # MPPCA: obtain MPPCA model parameters for the joint model and data
                                                                                                 
        self.pi, self.mu, self.W, self.sigma2, self.L_joint = mppca_fit(train_data,  M, q, itr = niter, rand=None)

        
        print('mixture weights: pi = ',self.pi)
#         print('mixture mu = ',self.mu)
        print('Log likelihood = ',self.L_joint[-1])
        plt.plot(self.L_joint)
        plt.ylabel('Loglikelihood', fontsize=12), plt.xlabel('Iter', fontsize=12)
        plt.show()
    def post_pdf_pri_m_dk(self):
        '''Function: Retrun posterior pdf f(m|d_k) for all d_k and Loglikelihood of joint f(m,d_k) for 
                        prior m samples (m_input), using MPPCA
            retrun:
            post_pdf_m_dk:posterior pdf f(m|d_k) for all d_k 
            L_joint: Loglikelihood of joint f(m,d_k)
        '''
        train_data = np.c_[self.m_input, self.d_input]
        dim_m = self.m_input.shape[1]
        
        N = train_data.shape[0]
        '''Step 1. Load MPPCA parameters'''
        pi, mu, W, sigma2, L_joint = self.pi, self.mu, self.W, self.sigma2, self.L_joint, 
        
        
        '''Step 2. Compute the joint pdf f(m, d_{k}) for all d_k using MPPCA'''
        # pos_m_d_all: N_p, N_d
        jnt_m_dk_all = np.zeros((N, N))
        for i in range(N):
            X = np.c_[self.m_input, np.repeat(train_data[i:i+1, dim_m:], N, axis=0)]
            jnt_m_dk_all[:,i] = self.mppca_pdf(X, pi, mu, W, sigma2)

        '''Step 3. MPPCA compute the data pdf $f(d_{k})$ for all d_k'''
        # print('Compute f(d).')
        #3.1 obtain MPPCA model parameters for data variable only
        pi, mu, W, sigma2, clusters = src.mppca.initialization_kmeans(self.d_input, self.M, self.q)
        pi, mu, W, sigma2, R, L_data, sigma2hist = src.mppca.mppca_gem(self.d_input, pi, mu, W, 
                                                                       sigma2, self.niter)
        # 3.2 calculate the $f(d_{k})$ for all d_k
        f_dk_all = np.zeros(N)
        for k in range(N):
            f_dk_all[k]= self.mppca_pdf(self.d_input[k:k+1, :], pi, mu, W, sigma2)
        # print('Compute f(m|d).')
        '''Step 4. Compute post pdf: $f(m|d_{k})= {f(m, d_{k})}/{f(d_{k})}$'''
        post_pdf_m_dk = jnt_m_dk_all/f_dk_all

        return post_pdf_m_dk, L_joint, f_dk_all

    def mppca_pdf(self, X, pi, mu, W, sigma2):
        '''Function to calculate pdf of sample X using the input MPPCA model: pi, mu, W, sigma2'''
        
        N, d = X.shape
        p = len(sigma2)
        
        # C: covar of mixture model
        C = np.zeros((p, d, d))
        C_inv = np.zeros((p, d, d))
        X_pdf = np.zeros(N)
        for i in range(p):

            C[i, :, :] = sigma2[i] * np.eye(d) + np.dot(W[i, :, :], W[i, :, :].T)

            # p_i(t)
            X_pdf_i = pi[i]*multivariate_normal.pdf(X , mean=mu[i, :], cov=C[i, :, :])

            # sum: p(t) = sum(p_i*p_i(t))
            X_pdf = X_pdf_i + X_pdf

        return X_pdf
    
    def post_m_dk_pdf(self, d_k, n_Pos_Samples, post_PDF=True, rand_seed=None):

        '''Function: Sample posterior f(m|d_k) and estimate corresponding posterior pdf with MPPCA
            Parameters:
                d_k: the k-th data variable or observed data variable, 2D array [1, dim_d]
                n_Pos_Samples: number of posterior samples, int. 
            Return:
                post_pdf_m_dk: samples of posterior f(m|d_k) from MPPCA
                mPos_Smpls_pdf: corresponding posterior PDF of posterior samples from MPPCA
                mixture_mu: the posterior mean of each mixture
            
            Reference: https://stats.stackexchange.com/questions/348941/general-conditional-distributions-for-multivariate-gaussian-mixtures
        '''

        dim_m, dim_d = self.m_input.shape[1], self.d_input.shape[1]
        dim_joint = dim_m+dim_d
        M = self.M
        q = self.q
        niter = self.niter
        n_sample = n_Pos_Samples



        '''Step 1. Train the MPPCA to estimate parameters'''
        # print('Compute f(m, d).')
        # MPPCA: obtain MPPCA model parameters for the joint model and data

        pi, mu, W, sigma2 = self.pi, self.mu, self.W, self.sigma2 


        '''Step 2.1 Calculate the covariance of posterior C_m|d for each mixture component'''
        #  C: Covariance of all Gaussian Mixtures of f(m,d)
        C = np.zeros((M, dim_joint, dim_joint))
        C_m_d_post = np.zeros((M, dim_m, dim_m))
        C_md = np.zeros((M, dim_m, dim_d))
        C_dd = np.zeros((M, dim_d, dim_d))
        C_dd_inv = np.zeros((M, dim_d, dim_d))

        for i in range(M):
            # compute the Covariance of i-th Gaussian Mixture of f(m,d)
            C_i = sigma2[i] * np.eye(dim_joint) + np.dot(W[i, :, :], W[i, :, :].T)
            # compute the C_mm, C_dd, C_md, C_dm, and C_dd_inverse of i-th Gaussian Mixture
            C_mm_i, C_dd_i = C_i[:dim_m, :dim_m], C_i[dim_m:, dim_m:]
            C_md_i, C_dm_i = C_i[:dim_m, dim_m:], C_i[dim_m:, :dim_m]
                                    
            
            if np.any(np.linalg.eigvals(C_dd_i)<0):
                C_dd_i = make_covariance_SPD(C_dd_i)
            
            C_dd_inv_i = np.linalg.inv(C_dd_i)

            #C_m|d (posterior) for each mixture component
            C_m_d_post_i = C_mm_i - C_md_i.dot(C_dd_inv_i).dot(C_dm_i)


            C[i, :, :] = np.copy(C_i)  
            C_md[i, :, :] = C_md_i
            C_dd[i, :, :] = C_dd_i
            C_dd_inv[i, :, :] = C_dd_inv_i
            C_m_d_post[i, :, :] = C_m_d_post_i

        '''Step 2.2 Calculate the posterior mean: u_m|d'''
        def mu_post(mu, d_k, dim_m, C_md, C_dd_inv):
            M = mu.shape[0]
            mu_post = np.zeros((M, dim_m))
            for i in range(M):
                # at each component, u_m|d = u_m + C_md*C_dd_inverse*((d_k - u_d).T)
                
                mu_post[i] = mu[i, :dim_m] + C_md[i].dot(C_dd_inv[i]).dot((d_k-mu[i:i+1,dim_m:]).T)[:,0]
            return mu_post
        
        mu_post = mu_post(mu, d_k, dim_m, C_md, C_dd_inv)
        '''Step 3. Sample f(m|d_{k}) from multivariate Gaussian mixtures, 
                    with mean as u_m|d (:mu_post)  and C_m|d (:C_m_d_post)'''  

        
        '''Step 3.1 Recalculate the mixture weights Pi_new (This is where the question occurs, 
                    should we use a new mixture model for d only, or the one derived from the joint?
                    Here I use the one derived from the joint)'''
        
        pdf_d_k = np.zeros(M)

        for i in range(M):
#             print(C_dd[i])
            pdf_d_k[i] = pi[i]*multivariate_normal.pdf(d_k , 
                                                       mean=mu[i,dim_m:], 
                                                       cov=C_dd[i])
            
        pi_new = pdf_d_k/pdf_d_k.sum()
        
        '''step 3.2. Generate n_sample random numbers from a categorical 
                     distribution of size M and probabilities pi '''
        np.random.seed(rand_seed)
        categr_N = np.random.choice(np.arange(M), size = n_Pos_Samples, p=pi_new)

        '''step 3.3 For mixture component i, 
                    generate n_sample_i random samples 
                    from the Norm_i((mu_m|d)_i,( C_m|d)_i).'''
        mPos_Smpls = np.zeros((n_sample, dim_m))
        i_start = 0
        mixture_mu = []
        for i in range(M):
            n_sample_i = (categr_N==i).sum()
            mPos_Smpls[i_start:i_start+n_sample_i] = multivariate_normal.rvs(mean=mu_post[i], 
                                                                             cov=C_m_d_post[i], 
                                                                             size=n_sample_i).reshape(-1, dim_m)
            '''calculate the mean of each mixture'''
            mixture_mu.append(mu_post[i])
            i_start += n_sample_i 
        mixture_mu = np.asarray(mixture_mu)


        if post_PDF:
            '''step 3.4 Estimate the posterior sample PDF using the gaussian mixtures'''
            mPos_Smpls_pdf = np.zeros(n_sample)    
            for i in range(M):
                # p_i(m|d_k)
                mPos_Smpls_pdf_i = pi_new[i]*multivariate_normal.pdf(mPos_Smpls , 
                                                                     mean=mu_post[i], 
                                                                     cov=C_m_d_post[i])
                # sum: p(m|d_k) = sum(p_i* p_i(m|d_k))
                mPos_Smpls_pdf = mPos_Smpls_pdf_i + mPos_Smpls_pdf
            return mPos_Smpls, mPos_Smpls_pdf, mixture_mu 
        else:        
            return mPos_Smpls, np.zeros(n_sample), mixture_mu  
    
    def smpl_pri_pdf(self, pri_m, m_samples, m_M, m_q):
        '''Main function: calcualte m_samples' prior PDF 
            Parameters:
                pri_m: data of prior m samples, 2D array [n_prior_samples, dim_m]
                m_samples: data of m_samples, 2D array [n_samples, dim_m]
                m_M: number of m's mixture components, int
                m_q: number of m's latent variables, int
            retrun:
                Smpls_pri_pdf: m_samples' prior PDF
                
                '''
        dim_m = pri_m.shape[1]
        endog_dim = 'c'*(dim_m)

        if dim_m==1:
            '''Compute prior pdf of m_samples with KDE'''
            # print('Variable has only 1 dim. Compute prior f(m) with KDE.')
            pri_kde = sm.nonparametric.KDEMultivariate(data=pri_m, var_type=endog_dim, bw='normal_reference')

            Smpls_pri_pdf = pri_kde.pdf(data_predict = m_samples)

        if dim_m>1: 
            '''Compute prior pdf of m_samples with Gaussian Mixture'''
            N = pri_m.shape[0]
            # print('Variable has more than 1 dim. Compute prior f(m) with MPPCA.')

            pi, mu, W, sigma2, clusters = src.mppca.initialization_kmeans(pri_m, m_M, m_q)
            # EM iteration
            pi, mu, W, sigma2, R, L_joint, sigma2hist = src.mppca.mppca_gem(pri_m, 
                                                                            pi, mu, W, sigma2, self.niter)    

            '''Calculate ovariance of all Gaussian Mixtures of f(m)'''
            C = np.zeros((m_M, dim_m, dim_m))

            for i in range(m_M):
                # compute the Covariance of i-th Gaussian Mixture of f(m,d)
                C[i, :, :] = sigma2[i] * np.eye(dim_m) + np.dot(W[i, :, :], W[i, :, :].T)


            '''Estimate the m_samples' prior PDF using the gaussian mixtures'''
            Smpls_pri_pdf = np.zeros(m_samples.shape[0])    
            for i in range(m_M):
                # p_i(m|d_k)
                Smpls_pri_pdf_i = pi[i]*multivariate_normal.pdf(m_samples, mean=mu[i], cov=C[i])
                # sum: p(m|d_k) = sum(p_i* p_i(m|d_k))
                Smpls_pri_pdf = Smpls_pri_pdf_i + Smpls_pri_pdf

        return Smpls_pri_pdf
    
    def mppca_rvs(self, n_samples, rand_seed=None):
        '''Random sample sample using the input MPPCA model: pi, mu, W, sigma2'''
        
        M = self.M
        q = self.q
        
        train_data = np.c_[self.m_input, self.d_input]
        N, d = train_data.shape
        
        '''Step 1. Estimaite the covariance matrix of mixtures'''
        
        # C: covar of mixture model
        C = np.zeros((M, d, d))
        
        for i in range(M):
            C[i, :, :] = self.sigma2[i] * np.eye(d) + np.dot(self.W[i, :, :], self.W[i, :, :].T)
    
        '''step 2.1 Generate n_sample random numbers from a categorical 
                     distribution of size M and probabilities pi '''
        np.random.seed(rand_seed)
        categr_N = np.random.choice(np.arange(M), size=n_samples, p=self.pi)
            
        '''step 2.2 For mixture component i, 
            generate n_sample_i random samples 
            from the Norm_i(mu_i,C_i).'''
        rvs_Smpls = np.zeros((n_samples, d))
        
        i_start = 0
        for i in range(M):
            n_sample_i = (categr_N==i).sum()
            rvs_Smpls[i_start:i_start+n_sample_i] = multivariate_normal.rvs(mean=self.mu[i], 
                                                                             cov=C[i], 
                                                                             size=n_sample_i).reshape(-1, d)
            i_start += n_sample_i
            
        '''Randomize the samples'''
        rand_index = np.arange(n_samples)
        np.random.seed(rand_seed)
        np.random.shuffle(rand_index)
        
        return rvs_Smpls[rand_index]