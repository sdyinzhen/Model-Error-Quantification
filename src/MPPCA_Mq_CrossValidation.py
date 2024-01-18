from src.mppca_vDavid import *
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
def MPPCA_Mq_CrossValidation(X, Mmin, Mmax, qmin, qmax,deltaq=1, k=10, EM_itr = 40):
    # yinzhen@stanford.edu; Mar 17, 2022
    # % X: samples (Nbmodels x dim)
    # % Mmin, Mmax: minimum and maximum number of mixtures to test
    # % qmin, Mmax: minimum dimension to test
    # % deltaq: steps size for q
    # k: k-fold cross validate, default k=10
    # EM_itr: EM interation numbers in MPPCA fitting
    
    # Return:
    # L_train, L_test: log-likelihood of training and testing set under different M,q; and plot.
    # Mq_opt: optimal [M,q]
    NbModels = X.shape[0]
    PermutedIdx = np.random.permutation(NbModels)
    GroupSize = np.round(NbModels/k).astype(int) # size of testing samples
    N_train = int(NbModels-GroupSize) # size of training samples
    N_q = len(np.arange(qmin, qmax, deltaq)) # number of tested q.
    L_test = np.zeros((k, Mmax-Mmin,N_q ))
    L_train = np.zeros((k, Mmax-Mmin,N_q ))
    # perform 10 fold cross-validation
    for i in tqdm(range(k)):
        
        # % Create the testing and training set
        idxTesting = PermutedIdx[(GroupSize*i):(GroupSize*(i+1))]
        TestingSet = X[idxTesting,:] # testing set
        
        idxTraining = np.setdiff1d(PermutedIdx, idxTesting, assume_unique=True)        
        TrainingSet = X[idxTraining,:] # training set
        
        for M_i in range(Mmin, Mmax):
            q = 0
            for q_i in range(qmin, qmax, deltaq):
                # train MPPCA
                pi, mu, W, sigma2, L_joint = mppca_fit(TrainingSet,  M=M_i, 
                                                       q=q_i, itr = EM_itr, rand=None)
                L_train[i, int(M_i-Mmin), q] = L_joint[-1]/N_train
                
                # loglikelihood of testing
                L_test_i = EvaluateLikelihood(TestingSet, pi, mu, W, sigma2)
                
                L_test[i, int(M_i-Mmin), q] = L_test_i/(NbModels-N_train)
                q = q+1
    # replace infinite values by nan
    L_train[L_train==-np.inf] = np.nan
    L_test[L_test==-np.inf] = np.nan
    # Average the likelihoods over the 10 folds. 
    L_train = np.nanmean(L_train, axis=0)
    L_test = np.nanmean(L_test, axis=0)
    
    # Optimal M and q
    q_s = np.arange(qmin, qmax, deltaq)
    M_s = np.arange(Mmin, Mmax)
    # 
    Mq_opt = [M_s[np.argwhere(L_test==L_test.max())[0][0]], 
              q_s[np.argwhere(L_test==L_test.max())[0][1]]]
    print('optimal M = ',Mq_opt[0], ', q = ',Mq_opt[1])
    # plot the loglikelihood
    plt.figure(figsize=(12,11))
    plt.subplot(121)
    plt.imshow(L_train, origin='lower')
    plt.colorbar(shrink=0.25)
    plt.title('LogLikelihood, training set')
    plt.ylabel('# of Mixtures M'), plt.xlabel('# of compoments q'), 
    plt.xticks(np.arange(L_train.shape[1]), q_s)
    plt.yticks(np.arange(L_train.shape[0]), M_s)
    
    plt.subplot(122)
    plt.imshow(L_test, origin='lower')
    plt.colorbar(shrink=0.25)
    plt.title('LogLikelihood, testing set')
    plt.ylabel('# of Mixtures M'), plt.xlabel('# of compoments q'), 
    plt.xticks(np.arange(L_test.shape[1]), q_s)
    plt.yticks(np.arange(L_test.shape[0]), M_s)
    plt.tight_layout()

    return L_train, L_test, Mq_opt