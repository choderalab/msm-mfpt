import pyemma
import numpy as np
import sklearn
import sklearn.model_selection
import multiprocessing
from glob import glob

X = pyemma.coordinates.load(glob("NTL9-*-protein-*.npy"), stride=50)

splits = np.load('ntl9_vamp_splits.npy')

def score_vamp(featurized_trajs, splits, tica_lag, tica_dim, microstate_number, msm_lag):

    scores = []

    for split in splits:

        if tica_dim == '95p':
            tica_kinetic = pyemma.coordinates.tica(list(np.array(featurized_trajs)[split[0]]), lag=tica_lag, var_cutoff=1, kinetic_map=True)
            dim_kinetic = np.argwhere(np.cumsum(tica_kinetic.eigenvalues**2)/np.sum(tica_kinetic.eigenvalues**2) > 0.95)[0,0] + 1
            Y_kinetic = tica_kinetic.transform(featurized_trajs)
            Y_kinetic = [traj[:,:dim_kinetic] for traj in Y_kinetic]
            
            tica_commute = pyemma.coordinates.tica(list(np.array(featurized_trajs)[split[0]]), lag=tica_lag, var_cutoff=1, kinetic_map=False, commute_map=True)
            dim_commute = np.argwhere(np.cumsum(tica_commute.timescales)/np.sum(tica_commute.timescales) > 0.95)[0,0] + 1
            Y_commute = tica_commute.transform(featurized_trajs)
            Y_commute = [traj[:,:dim_commute] for traj in Y_commute]
        else:
            tica_kinetic = pyemma.coordinates.tica(featurized_trajs, lag=tica_lag, var_cutoff=1, kinetic_map=True)
            Y_kinetic = tica_kinetic.get_output()
            Y_kinetic = [traj[:,:tica_dim] for traj in Y_kinetic]
            
            tica_commute = pyemma.coordinates.tica(featurized_trajs, lag=tica_lag, var_cutoff=1, kinetic_map=False, commute_map=True)
            Y_commute = tica_commute.get_output()
            Y_commute = [traj[:,:tica_dim] for traj in Y_commute]
        
        kmeans_kinetic = pyemma.coordinates.cluster_kmeans(list(np.array(Y_kinetic)[split[0]]), k=microstate_number, max_iter=1000, n_jobs=1)
        dtrajs_kinetic_train = kmeans_kinetic.dtrajs
        dtrajs_kinetic_test = kmeans_kinetic.transform(list(np.array(Y_kinetic)[split[1]]))
        dtrajs_kinetic_test = [np.concatenate(traj) for traj in dtrajs_kinetic_test]
    
        kmeans_commute = pyemma.coordinates.cluster_kmeans(list(np.array(Y_commute)[split[0]]), k=microstate_number, max_iter=1000, n_jobs=1)
        dtrajs_commute_train = kmeans_commute.dtrajs
        dtrajs_commute_test = kmeans_commute.transform(list(np.array(Y_commute)[split[1]]))
        dtrajs_commute_test = [np.concatenate(traj) for traj in dtrajs_commute_test]
        
        # we're doing only VAMP-2 and automatically selecting the max. number of eigenvalues later
        # for now have to enumerate as many k's as possible - let's go every 1 up to min number of clusters in any model
        msm_kinetic = pyemma.msm.estimate_markov_model(dtrajs_kinetic_train, msm_lag, score_method='VAMP2')
        msm_commute = pyemma.msm.estimate_markov_model(dtrajs_commute_train, msm_lag, score_method='VAMP2')
        
        if tica_dim == '95p':
            for score_k in np.arange(2,51):
                score_kinetic = msm_kinetic.score(dtrajs_kinetic_test, score_method='VAMP2', score_k=score_k)
                score_kinetic_train = msm_kinetic.score(dtrajs_kinetic_train, score_method='VAMP2', score_k=score_k)
                score_commute = msm_commute.score(dtrajs_commute_test, score_method='VAMP2', score_k=score_k)
                score_commute_train = msm_commute.score(dtrajs_commute_train, score_method='VAMP2', score_k=score_k)
                scores.append([(score_kinetic, score_commute, score_kinetic_train, score_commute_train), (dim_kinetic, dim_commute)])
        else:
            for score_k in np.arange(2,51):
                score_kinetic = msm_kinetic.score(dtrajs_kinetic_test, score_method='VAMP2', score_k=score_k)
                score_kinetic_train = msm_kinetic.score(dtrajs_kinetic_train, score_method='VAMP2', score_k=score_k)
                score_commute = msm_commute.score(dtrajs_commute_test, score_method='VAMP2', score_k=score_k)
                score_commute_train = msm_commute.score(dtrajs_commute_train, score_method='VAMP2', score_k=score_k)
                scores.append((score_kinetic, score_commute, score_kinetic_train, score_commute_train))
    
    return scores

def score_vamp_multiprocess(featurized_trajs, splits, tica_lags, tica_dims, microstate_numbers, msm_lag, threads=8):

    parameters = []
    parameters_return = []

    for tica_lag in tica_lags:
        for tica_dim in tica_dims:
            for microstate_number in microstate_numbers:
                parameters.append((featurized_trajs, splits, tica_lag, tica_dim, microstate_number, msm_lag))
                parameters_return.append((tica_lag, tica_dim, microstate_number, msm_lag))

    pool = multiprocessing.Pool(threads)
    scores = pool.starmap(score_vamp, parameters)
    return [parameters_return, scores]

# prepare the parameter choices - use msm_lag = 50 (50 x 0.2 ns/frame = 10 ns)
# 10-strided lag time 1, i.e. 50 frames non-strided
tica_lags = [1]
tica_dims = ['95p'] + [2,10,50,100,300,500]
microstate_numbers = [50,100,200,400,600,800,1000]

# 50-strided lag time 1, i.e. 50 frames non-strided
scores = score_vamp_multiprocess(X, splits, tica_lags, tica_dims, microstate_numbers, 1, 32)

np.save('scores_msmlag10ns_splittica_alleigen', scores)
