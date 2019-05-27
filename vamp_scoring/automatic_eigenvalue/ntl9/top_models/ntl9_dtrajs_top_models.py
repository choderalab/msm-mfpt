# before running set:
# export PYEMMA_NJOBS=1 OMP_NUM_THREADS=1

import pyemma
import numpy as np
from glob import glob
import scipy
import multiprocessing

# number of threads
n_proc = 9

models = ["('commute', 1, 10, 100)",
 "('commute', 1, 10, 200)",
 "('commute', 1, 10, 50)",
 "('commute', 1, 10, 600)",
 "('commute', 1, 2, 200)",
 "('commute', 1, 2, 600)",
 "('commute', 1, 50, 100)",
 "('commute', 1, 50, 200)",
 "('kinetic', 1, 10, 200)"]
 
traj_files1 = sorted(glob('NTL9-0*.npy'))
traj_files2 = sorted(glob('NTL9-2*.npy'))
traj_files3 = sorted(glob('NTL9-3*.npy'))

source = pyemma.coordinates.source([traj_files1, traj_files2, traj_files3], chunksize=10000)
X = source.get_output()
 
def get_dtrajs(model):
    # read the hyperparameters
    split = model.split(', ')
    map_ = split[0][1:]
    lag = int(split[1]) * 50
    dim = int(split[2])
    states = int(split[3][:-1])
    
    if map_ == "'kinetic'":
        tica = pyemma.coordinates.tica(X, lag=lag, dim=dim, kinetic_map=True)
    elif map_ == "'commute'":    
        tica = pyemma.coordinates.tica(X, lag=lag, dim=dim, kinetic_map=False, commute_map=True)
    else:
        raise Exception('Bad tICA map type')    
    
    Y = tica.get_output()
    
    kmeans = pyemma.coordinates.cluster_kmeans(Y, k=states, max_iter=1000)
    
    dtrajs = kmeans.dtrajs
    
    return dtrajs
    

pool = multiprocessing.Pool(n_proc)

all_dtrajs = pool.map(get_dtrajs, models)

np.save('dtrajs_top_models', all_dtrajs)
