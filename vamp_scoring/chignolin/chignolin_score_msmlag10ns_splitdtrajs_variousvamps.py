import pyemma
import numpy as np
import sklearn
import sklearn.model_selection
import multiprocessing
from glob import glob

traj_files = glob("*.dcd")
pdb_file = "chig_pdb_166.pdb"

features = pyemma.coordinates.featurizer(pdb_file)
features.add_residue_mindist()
source = pyemma.coordinates.source(traj_files, features=features, chunk_size=10000)
X = source.get_output()

splits = np.load('chignolin_vamp1_splits.npy')

def score_vamp1(featurized_trajs, splits, tica_lag, tica_dim, microstate_number, msm_lag):

    scores = []

    for split in splits:

        if tica_dim == '95p':
            tica_kinetic = pyemma.coordinates.tica(featurized_trajs, lag=tica_lag, var_cutoff=1, kinetic_map=True)
            dim_kinetic = np.argwhere(np.cumsum(tica_kinetic.eigenvalues**2)/np.sum(tica_kinetic.eigenvalues**2) > 0.95)[0,0] + 1
            Y_kinetic = tica_kinetic.get_output()
            Y_kinetic = [traj[:,:dim_kinetic] for traj in Y_kinetic]
    
            tica_commute = pyemma.coordinates.tica(featurized_trajs, lag=tica_lag, var_cutoff=1, kinetic_map=False, commute_map=True)
            dim_commute = np.argwhere(np.cumsum(tica_commute.timescales)/np.sum(tica_commute.timescales) > 0.95)[0,0] + 1
            Y_commute = tica_commute.get_output()
            Y_commute = [traj[:,:dim_commute] for traj in Y_commute]
        else:
            tica_kinetic = pyemma.coordinates.tica(featurized_trajs, lag=tica_lag, var_cutoff=1, kinetic_map=True)
            Y_kinetic = tica_kinetic.get_output()
            Y_kinetic = [traj[:,:tica_dim] for traj in Y_kinetic]
    
            tica_commute = pyemma.coordinates.tica(featurized_trajs, lag=tica_lag, var_cutoff=1, kinetic_map=False, commute_map=True)
            Y_commute = tica_commute.get_output()
            Y_commute = [traj[:,:tica_dim] for traj in Y_commute]

        kmeans_kinetic = pyemma.coordinates.cluster_kmeans(Y_kinetic, k=microstate_number, max_iter=1000, n_jobs=1)
        dtrajs_kinetic = kmeans_kinetic.dtrajs
        dtrajs_kinetic_train = list(np.array(dtrajs_kinetic)[split[0]])
        dtrajs_kinetic_test = list(np.array(dtrajs_kinetic)[split[1]])

        kmeans_commute = pyemma.coordinates.cluster_kmeans(Y_commute, k=microstate_number, max_iter=1000, n_jobs=1)
        dtrajs_commute = kmeans_commute.dtrajs
        dtrajs_commute_train = list(np.array(dtrajs_commute)[split[0]])
        dtrajs_commute_test = list(np.array(dtrajs_commute)[split[1]])

        # 5 eigenvalues
        score_k = 5
        msm_kinetic = pyemma.msm.estimate_markov_model(dtrajs_kinetic_train, msm_lag, score_method='VAMP1', score_k=score_k)
        score_kinetic_5 = msm_kinetic.score(dtrajs_kinetic_test, score_method='VAMP1', score_k=score_k)
        score_kinetic_train_5 = msm_kinetic.score(dtrajs_kinetic_train, score_method='VAMP1', score_k=score_k)

        msm_commute = pyemma.msm.estimate_markov_model(dtrajs_commute_train, msm_lag, score_method='VAMP1', score_k=score_k)
        score_commute_5 = msm_kinetic.score(dtrajs_commute_test, score_method='VAMP1', score_k=score_k)
        score_commute_train_5 = msm_kinetic.score(dtrajs_commute_train, score_method='VAMP1', score_k=score_k)

        # 10 eigenvalues
        score_k = 10
        msm_kinetic = pyemma.msm.estimate_markov_model(dtrajs_kinetic_train, msm_lag, score_method='VAMP1', score_k=score_k)
        score_kinetic_10 = msm_kinetic.score(dtrajs_kinetic_test, score_method='VAMP1', score_k=score_k)
        score_kinetic_train_10 = msm_kinetic.score(dtrajs_kinetic_train, score_method='VAMP1', score_k=score_k)

        msm_commute = pyemma.msm.estimate_markov_model(dtrajs_commute_train, msm_lag, score_method='VAMP1', score_k=score_k)
        score_commute_10 = msm_kinetic.score(dtrajs_commute_test, score_method='VAMP1', score_k=score_k)
        score_commute_train_10 = msm_kinetic.score(dtrajs_commute_train, score_method='VAMP1', score_k=score_k)

        # 10 eigenvalues
        score_k = 50
        msm_kinetic = pyemma.msm.estimate_markov_model(dtrajs_kinetic_train, msm_lag, score_method='VAMP1', score_k=score_k)
        score_kinetic_50 = msm_kinetic.score(dtrajs_kinetic_test, score_method='VAMP1', score_k=score_k)
        score_kinetic_train_50 = msm_kinetic.score(dtrajs_kinetic_train, score_method='VAMP1', score_k=score_k)
    
        msm_commute = pyemma.msm.estimate_markov_model(dtrajs_commute_train, msm_lag, score_method='VAMP1', score_k=score_k)
        score_commute_50 = msm_kinetic.score(dtrajs_commute_test, score_method='VAMP1', score_k=score_k)
        score_commute_train_50 = msm_kinetic.score(dtrajs_commute_train, score_method='VAMP1', score_k=score_k)

        # now VAMP-2

        # 5 eigenvalues
        score_k = 5
        msm_kinetic = pyemma.msm.estimate_markov_model(dtrajs_kinetic_train, msm_lag, score_method='VAMP2', score_k=score_k)
        score_kinetic_5_2 = msm_kinetic.score(dtrajs_kinetic_test, score_method='VAMP2', score_k=score_k)
        score_kinetic_train_5_2 = msm_kinetic.score(dtrajs_kinetic_train, score_method='VAMP2', score_k=score_k)

        msm_commute = pyemma.msm.estimate_markov_model(dtrajs_commute_train, msm_lag, score_method='VAMP2', score_k=score_k)
        score_commute_5_2 = msm_kinetic.score(dtrajs_commute_test, score_method='VAMP2', score_k=score_k)
        score_commute_train_5_2 = msm_kinetic.score(dtrajs_commute_train, score_method='VAMP2', score_k=score_k)

        # 10 eigenvalues
        score_k = 10
        msm_kinetic = pyemma.msm.estimate_markov_model(dtrajs_kinetic_train, msm_lag, score_method='VAMP2', score_k=score_k)
        score_kinetic_10_2 = msm_kinetic.score(dtrajs_kinetic_test, score_method='VAMP2', score_k=score_k)
        score_kinetic_train_10_2 = msm_kinetic.score(dtrajs_kinetic_train, score_method='VAMP2', score_k=score_k)

        msm_commute = pyemma.msm.estimate_markov_model(dtrajs_commute_train, msm_lag, score_method='VAMP2', score_k=score_k)
        score_commute_10_2 = msm_kinetic.score(dtrajs_commute_test, score_method='VAMP2', score_k=score_k)
        score_commute_train_10_2 = msm_kinetic.score(dtrajs_commute_train, score_method='VAMP2', score_k=score_k)

        # 10 eigenvalues
        score_k = 50
        msm_kinetic = pyemma.msm.estimate_markov_model(dtrajs_kinetic_train, msm_lag, score_method='VAMP2', score_k=score_k)
        score_kinetic_50_2 = msm_kinetic.score(dtrajs_kinetic_test, score_method='VAMP2', score_k=score_k)
        score_kinetic_train_50_2 = msm_kinetic.score(dtrajs_kinetic_train, score_method='VAMP2', score_k=score_k)

        msm_commute = pyemma.msm.estimate_markov_model(dtrajs_commute_train, msm_lag, score_method='VAMP2', score_k=score_k)
        score_commute_50_2 = msm_kinetic.score(dtrajs_commute_test, score_method='VAMP2', score_k=score_k)
        score_commute_train_50_2 = msm_kinetic.score(dtrajs_commute_train, score_method='VAMP2', score_k=score_k)

        if tica_dim == '95p':
            scores.append([(score_kinetic_5, score_commute_5, score_kinetic_train_5, score_commute_train_5), (dim_kinetic, dim_commute)])
            scores.append([(score_kinetic_10, score_commute_10, score_kinetic_train_10, score_commute_train_10), (dim_kinetic, dim_commute)])
            scores.append([(score_kinetic_50, score_commute_50, score_kinetic_train_50, score_commute_train_50), (dim_kinetic, dim_commute)])
            scores.append([(score_kinetic_5_2, score_commute_5_2, score_kinetic_train_5_2, score_commute_train_5_2), (dim_kinetic, dim_commute)])
            scores.append([(score_kinetic_10_2, score_commute_10_2, score_kinetic_train_10_2, score_commute_train_10_2), (dim_kinetic, dim_commute)])
            scores.append([(score_kinetic_50_2, score_commute_50_2, score_kinetic_train_50_2, score_commute_train_50_2), (dim_kinetic, dim_commute)])
        else:
            scores.append((score_kinetic_5, score_commute_5, score_kinetic_train_5, score_commute_train_5))
            scores.append((score_kinetic_10, score_commute_10, score_kinetic_train_10, score_commute_train_10))
            scores.append((score_kinetic_50, score_commute_50, score_kinetic_train_50, score_commute_train_50))
            scores.append((score_kinetic_5_2, score_commute_5_2, score_kinetic_train_5_2, score_commute_train_5_2))
            scores.append((score_kinetic_10_2, score_commute_10_2, score_kinetic_train_10_2, score_commute_train_10_2))
            scores.append((score_kinetic_50_2, score_commute_50_2, score_kinetic_train_50_2, score_commute_train_50_2))
    return scores

def score_vamp1_multiprocess(featurized_trajs, splits, tica_lags, tica_dims, microstate_numbers, msm_lag, threads=8):

    parameters = []
    parameters_return = []

    for tica_lag in tica_lags:
        for tica_dim in tica_dims:
            for microstate_number in microstate_numbers:
                parameters.append((featurized_trajs, splits, tica_lag, tica_dim, microstate_number, msm_lag))
                parameters_return.append((tica_lag, tica_dim, microstate_number, msm_lag))

    pool = multiprocessing.Pool(threads)
    scores = pool.starmap(score_vamp1, parameters)
    return [parameters_return, scores]

# prepare the parameter choices - use msm_lag = 50 (50 x 0.2 ns/frame = 10 ns), score_k = 5
tica_lags = [5,25,50]
#tica_dims = ['95p'] + [2,10,20,28]
tica_dims = ['95p', 2]
microstate_numbers = [5,10,50] + list(np.arange(100,1100,100))

scores = score_vamp1_multiprocess(X, splits, tica_lags, tica_dims, microstate_numbers, 50, 32)

np.save('scores_msmlag10ns_splitdtrajs_variousvamps', scores)
