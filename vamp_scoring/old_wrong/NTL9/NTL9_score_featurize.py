import pyemma
import multiprocessing
from glob import glob
import numpy as np

traj_files = glob("*.dcd")
pdb_file = "NTL9.pdb"

features = pyemma.coordinates.featurizer(pdb_file)
features.add_residue_mindist()

def featurize(traj):
    source = pyemma.coordinates.source(traj, features=features)
    X = source.get_output()
    np.save('%s' % traj[:-4], X[0])

pool = multiprocessing.Pool(32)
pool.map(featurize, traj_files)
