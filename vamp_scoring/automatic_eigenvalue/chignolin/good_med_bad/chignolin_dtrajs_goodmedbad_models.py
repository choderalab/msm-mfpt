# before running set:
# export PYEMMA_NJOBS=1
# export OMP_NUM_THREADS=1

import pyemma
import numpy as np
from glob import glob
import scipy
import multiprocessing

# number of threads
n_proc = 10

models = ["('kinetic', 5, 15, 100)", "('kinetic', 25, 6, 600)",
       "('kinetic', 50, 2, 200)", "('kinetic', 50, 2, 500)",
       "('kinetic', 50, 3, 900)", "('kinetic', 25, 6, 900)",
       "('commute', 5, 2, 400)", "('commute', 5, 2, 50)",
       "('commute', 50, 2, 600)", "('commute', 50, 23, 1000)"]
 
pdb_file = "chig_pdb_166.pdb"
traj_files = sorted(glob("*.dcd"))

features = pyemma.coordinates.featurizer(pdb_file)
features.add_residue_mindist()
source = pyemma.coordinates.source([traj_files], features=features, chunksize=10000)
X = source.get_output()
 
def get_dtrajs(model):
    # read the hyperparameters
    split = model.split(', ')
    map_ = split[0][1:]
    lag = int(split[1])
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

np.save('dtrajs_goodmedbad_models', all_dtrajs)
