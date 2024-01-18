from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
import scipy 
import numpy as np
from src.mppca_vDavid import *
# David Zhen Yin, of the MPPCA algorithm described in
# "Mixtures of Probabilistic Principal Component Analysers",
# Michael E. Tipping and Christopher M. Bishop, Neural Computation 11(2),
# pp 443–482, MIT Press, 1999

def mppca_fit(tn, M, q, itr = 30, rand=None):
    '''This is the function to fit a MPPCA model to a high dim data sample tn.
        Including initialization, EM iteration
        Parameters:
        tn: input data sample, 2d array, [n_samples, n_dim]
        M: number of mixture components, int
        q: number of latent space, int
        itr: number of EM iterations, int
        rand: random initialization seeds, int. 
    
    '''
    
    
    '''1. initialization'''
    pi_ini, mu_ini, W_ini, sigma2_ini, clusters_ini, Rni_ini = initialization_kmeans(tn, 
                                                                                     M, q,
                                                                                     cluster_rnd=rand)
    '''2. Apply the EM iterative procedure'''
    # I: EM iteration steps
    I = itr
    # LogL: log-likelihood
    LogL = np.zeros(I+1)

    '''2.1 initial M-step'''
    pi_new, mu_new, W_new, sigma2_new = UpdateParams_M_step(tn, 
                                                            pi_ini, 
                                                            mu_ini, 
                                                            W_ini, sigma2_ini, Rni_ini)

    LogL[0] = EvaluateLikelihood(tn, pi_new, mu_new, W_new, sigma2_new)
    for i in range(I):
        '''2.2 E-Step'''
        Rni_new = PosteriorResponsability(tn, pi_new, mu_new, W_new, sigma2_new)
        '''2.3 M-Step'''
        pi_new, mu_new, W_new, sigma2_new = UpdateParams_M_step(tn, pi_new, 
                                                                mu_new, 
                                                                W_new, 
                                                                sigma2_new, 
                                                                Rni_new)
        LogL[i+1] = EvaluateLikelihood(tn, pi_new, mu_new, W_new, sigma2_new)
    return pi_new, mu_new, W_new, sigma2_new, LogL

def mppca_pdf(Xpred, pi, mu, W, sigma2):
    '''Calculate the pdf of data sample Xpred using the fited mppca model'''
    pdf_Xpred = np.zeros(Xpred.shape[0])
    M = pi.shape[0]
    for i in range(M):
        pdf_Xpred = pdf_Xpred + pi[i]*DensityComponenti(Xpred, mu[i], W[i], sigma2[i])
    return pdf_Xpred

def initialization_kmeans(tn, M, q, cluster_rnd = None):
    """
    Initialize with PCA
    Author: David Zhen Yin: yinzhen@stanford.edu
    tn : sample dataset, 2D array [N_samples, d_dim]
    M : number of clusters (number of mixture components), int
    q : dimension of the latent space, int
    cluster_rnd: random state of k-mean cluster, int
    Return 
    pi : proportions of clusters
    mu : centers of the clusters in the observation space
    W : latent to observation matricies
    sigma2 : noise
    """

    N, d = tn.shape


    min_cls = 1
    itr = 1
    ''' Kmean clustering to partition mixtures
        and make sure each cluster has >1 samples'''
    while min_cls ==1:
        kmeans = KMeans(n_clusters=M, random_state=cluster_rnd).fit(tn)
        clusters = kmeans.labels_
        cls_size = np.zeros(M)
        for c in range(M):
            cls_size[c] = (clusters==c).sum()
        min_cls = cls_size.min()
        itr = itr + 1
        if itr>1000:
            break
            
    '''Initialize pi, mu, W, and sigma2 with PCA for each clustered component'''
    pi = np.zeros(M)
    mu = np.zeros((M, d))
    Rni = np.zeros((M, N))
    W = np.zeros((M, d, q))
    sigma2 = np.zeros(M)
    for c in range(M):
        
        if (clusters == c).sum() == 1:
            pi[c] = 0
            
            mu[c, :] = 0
            W[c, :] = 0
            sigma2[c, :] = 0
#             Rni{i} = (idx == i);
        else:
        
            tn_c = np.copy(tn[clusters == c, :])
            pi[c] = (clusters == c).sum()/N
            mu[c, :] = tn_c.mean(axis=0)
            # Perform PCA for each clustered mixture component
            pca = PCA(svd_solver='full').fit(tn_c)
            
            if len(pca.explained_variance_)>=q:
                # eigen values: diagonal
                lambdas = np.diag(pca.explained_variance_)
                # eigen vectors: [n_components, n_pcs]
                Us = pca.components_.T
            else:
                # if one cluster have samples < q
                n_dim, n_pcs = pca.components_.T.shape
                lambdas = np.diag(np.r_[pca.explained_variance_,np.zeros(q-n_pcs)])
                Us = np.c_[pca.components_.T, np.zeros((n_dim, q-n_pcs))]

            # Noise variance , eq.3.13 of the paper.
            sigML = (1/(d-q))*sum(np.diag(lambdas[q:d,q:d]))
            # weight matrix W. eq.3.12 of the paper.
            WML = Us[:,:q].dot(np.sqrt(lambdas[:q,:q] - sigML*np.eye(q)))
            # Assign W and and Noise variance to the corresponding mixture
            W[c] = WML
            sigma2[c] = sigML
            Rni[c] = clusters == c

    return pi, mu, W, sigma2, clusters, Rni


def UpdateParams_M_step(tn, pi_ini, mu_ini, W_ini, sigma2_ini, Rni):
    '''The M-step involves maximizing equation C.5 and 
        obtain new MPPCA parameter values.'''
    # number of samples (N) and dimension (d)
    N, d = tn.shape
    # number of mixtures
    M = W_ini.shape[0]
    # number of latent space
    q = W_ini.shape[2]
    pi_new = np.zeros(pi_ini.shape)
    mu_new = np.zeros(mu_ini.shape)
    W_new = np.zeros(W_ini.shape)
    sigma2_new = np.zeros(sigma2_ini.shape)
    for i in range(M):
        '''first EM step: update pi and mu'''
        pi_new[i] = (1/N)*sum(Rni[i]); 
        mu_new[i] = (Rni[i]*tn.T).sum(axis=1)/np.sum(Rni[i])
        
        '''second EM step: update Wi amd Sigma, eq.4.6'''
        Si = (1/(pi_new[i]*N))*(Rni[i]*(tn-mu_new[i]).T).dot(tn-mu_new[i])
        # NB scipy.eigh is the same as matlab
        lambdas,U = scipy.linalg.eigh(Si)
        #Lamda is in ascending order
        if np.any(np.iscomplex(lambdas)):
            '''If some of the eigen values are negative, make Si symmetric '''
            Si = (Si + Si.T)/2
            lambdas,U = scipy.linalg.eigh(Si)
        # Sort eigenval and eigenvec in descending order
        srt_idx = lambdas.argsort()[::-1]   
        lambdas = lambdas[srt_idx]
        Us = U[:,srt_idx]
        # Make eigen values as diagonal matrix
        lambdas = np.diag(lambdas)

        # Noise variance , eq.3.13 of the paper.
        sigML = (1/(d-q))*sum(np.diag(lambdas[q:d,q:d]))
        # weight matrix W. eq.3.12 of the paper.
        WML = Us[:,:q].dot(np.sqrt(lambdas[:q,:q] - sigML*np.eye(q)))
        # Assign W and and Noise variance to the corresponding mixture
        W_new[i] = WML
        sigma2_new[i] = sigML
    return  pi_new, mu_new, W_new, sigma2_new

def PosteriorResponsability(tn, pi, mu, W, sigma2):
    '''Eq 4.3： Compute all posterior responsabilities for the the M components.
        Fist iteration use kmeans, then use the current parameter'''
    N = tn.shape[0]
    M = pi.shape[0]
    Rni = np.zeros((M, N))
    
    # calclate probability p(t): pt and p(t|i): pti
    pt = np.zeros(N)
    pti = np.zeros((M, N))
    for i in range(M):
        pti[i] = DensityComponenti(tn, mu[i], W[i], sigma2[i])
        pt = pt + pi[i]*pti[i]
    # calculate posterior responsibility
    for i in range(M):
         Rni[i] = (pti[i]*pi[i])/pt;
    return Rni

def EvaluateLikelihood(tn, pi, mu, W, sigma2, Rni=np.nan):
    '''Evaluate log likelihood of the MPPCA model: Eq 4.2 or Eq C.12 (LC)'''
    N = tn.shape[0]
    M = pi.shape[0]

    pt = np.zeros(N)
    if np.all(np.isnan(Rni)):
        #  Eq 4.2
        for i in range(M):
            pt = pt + pi[i]*DensityComponenti(tn, mu[i], W[i], sigma2[i])
        L = np.sum(np.log(pt))
    else: 
        # Eq C.12 (LC)
        for i in range(M):
            pt_i = DensityComponenti(tn, mu[i], W[i], sigma2[i])
            # avoid log(0)
            pt_i[pt_i==0]=1e-23
            pt = pt + Rni[i]*np.log(pi[i]*pt_i)
        L = np.sum(pt)
    return L

def DensityComponenti(t, mu_i, W_i, sigma2_i):
    '''Estimate data samples density under i-th component. Eq. 5.1
        t: input data sample to evaluate the pdf, 2D array [n-samples, d_dim]
        W_i, sigma2_i, mu_i: MPPCA parameters of of one component
        '''
    d = t.shape[1]
    C_i = sigma2_i * np.eye(d) + np.dot(W_i[:, :], W_i[:, :].T)
    if np.all(np.linalg.eigvals(C_i)>0):
        pti = multivariate_normal.pdf(t, mean=mu_i, cov=C_i)
    else:
        '''The covariance matrix C_i is not symetric symmetric and positive definite'''
        C_i_SPD = make_covariance_SPD(C_i)
        pti = multivariate_normal.pdf(t, mean=mu_i, cov=C_i_SPD)
    return pti




def make_covariance_SPD(C_not_spd, eigval_TH = 1e-6):
    '''Make non-SPD covariance matrix Symmetric and Positive Definite'''
    # Make it symmetric
    C_S = (C_not_spd+C_not_spd.T)/2
    # Calculate the eigendecomposition of your matrix (A = V*D*V')
    Dc, Vc=scipy.linalg.eigh(C_S)
    # Set any eigenvalues that are lower than threshold "TH" ("TH" here being
    # equal to 1e-6) to a fixed non-zero "small" value (here assumed equal to 1e-6
    Dc[Dc <= eigval_TH] = eigval_TH
    # Built the "corrected" diagonal eigval matrix "D_c"
    Dc = np.diag(Dc)
    # Recalculate your matrix in its SPD variant "C_SPD"
    C_SPD = Vc.dot(Dc).dot(Vc.T)
    return C_SPD