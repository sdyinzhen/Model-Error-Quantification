
#Author: David Zhen Yin
#Contact: yinzhen@stanford.edu
#Date: September 11, 2018



import numpy as np
import matplotlib.pyplot as plt
import datetime

def scree_plot(input_data, data_name, keep_info_prcnt, plot=True):
    '''This is the PCA scree plot function, 
       This function also report the number of PCs that preserves the assigned amount of information of the input_data. 
       input_data: orignial input matrix for PCA analys; pc_num: number of pc components. 
       data_name: name of the input data, e.g. 'model', 'data'
       keep_info_prcnt: the amount of infomation (cumulative variance ratio) to preserve after PCA. 
       plot: 'True' - will produce the screet plot.
       '''
    X = input_data-input_data.mean(axis=0)
    eig_val, eig_vec = np.linalg.eig(X.dot(X.transpose()))
    eigval_sum = np.sum(eig_val)
    infor_list = np.cumsum(eig_val)/eigval_sum
    infor_list = np.array(np.where(infor_list<=keep_info_prcnt/100))[0]
    
    print('PC1-PC'+str(infor_list[-1]+1) + ' contain '+str(keep_info_prcnt) + '% of variance')

    if plot == True:
        cum_var_ratio =  np.cumsum(eig_val)/eigval_sum
        plt_pcs = len(cum_var_ratio[cum_var_ratio<0.998])+1
        
        plt.figure(figsize=(6, 4))
        plt.plot(np.arange(1, plt_pcs+1), cum_var_ratio[:plt_pcs], \
                 marker='o', markersize=5, linestyle = 'dashed', color='blue')
        plt.xticks(fontsize = 14)    
        plt.yticks(np.arange(0,1.01,0.1),fontsize = 14)
        plt.xlabel('number of PCs', fontsize = 12, weight='bold')
        plt.ylabel('cumulative variance ratio', fontsize = 12, weight='bold')
        plt.title('PCA scree plot of ' + data_name , fontsize=18, loc='left', style='italic')
        plt.grid(linestyle='dashed')
        plt.axhline(y=keep_info_prcnt/100, linewidth=2, color='red', linestyle='--')

        plt.axvline(x=infor_list[-1]+1, linewidth=2, color='red', linestyle='--')
    
    var_explained_by_pc = np.around((eig_val)/eigval_sum, 5)
    
    return var_explained_by_pc
	
	
## evd_fast(X, n_components)
## David Yin, yinzhen@stanford.edu
## Date: Oct 29, 2018

## This is the function to caculated eigen vectors for matrix with semi-large dimension, e.g.: matrix with dimension (LxP), where L<<P.
## X: input matrix with dimension LxP, where L<<P. 
## n_components: (int), the estimated number of eigen vectors (PC componets)
## example: Xeig_vecs = pca_fast(X, 10) will calculate the first 10 eigen vectors for matrix X. 


def evd_fast(X, n_components):
    print((datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')))
    X = X - X.mean(axis=0)
    eig_val, eig_vec = np.linalg.eig(X.dot(X.transpose()))
    new_eig_vecs = []
    for i in range(n_components):
        new_vec = X.transpose().dot(eig_vec[:,i:i+1])
        new_eig_vecs.append(new_vec[:,0]/np.linalg.norm(new_vec))
    new_eig_vecs = np.asarray(new_eig_vecs).T
    print((datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')))
    return new_eig_vecs
# This function returns the first eigen value of matrix X.
def first_eigval(X):
    X = X - X.mean(axis=0)
    eig_val, eig_vec = np.linalg.eig(X.dot(X.transpose()))
    return eig_val[0]

def eigen_imgs(eigen_vecs, eig_nums, i_dim,j_dim):
    '''
    This is the function to plot the eigen_images
    arg:
        eigen_vecs: the ndarray of the eigen vectors
        eig_nums: 1d arrary defines which pc numbers to plot
        i_dim, j_dim: the i and j dimension of the grid model        
    '''
    plot_num = len(eig_nums)
    fig_row = int((plot_num+3)/4)
    fig=plt.figure(figsize=(15, fig_row*3))
    
    count = 1
    for i in eig_nums:
        plot=fig.add_subplot(fig_row, 4, count)
        count = count+1
        plt.imshow(eigen_vecs[:,i-1].reshape(j_dim,i_dim), cmap='jet')       
        plt.xticks(fontsize = 13)
        plt.yticks(fontsize = 13)
        plt.title('model eigen_img (PC' + str(i) +')', fontsize = 14)
    plt.subplots_adjust(top=0.55, bottom=0.08, left=0.10, right=0.95, hspace=0.15,
                    wspace=0.35)
    
    #t = (" ")
    #plt.figure(figsize=(3, 0.1))
    #plt.text(0, 0, t, style='normal', ha='center', fontsize=16, weight = 'bold')
    #plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())
    #plt.show()
    return
