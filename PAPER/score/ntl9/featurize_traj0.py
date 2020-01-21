from glob import glob
import multiprocessing
import mdtraj as md
import numpy as np

def calc_contacts(traj):
    traj = md.load(traj, top='NTL9.pdb')
    contacts = md.compute_contacts(traj)[0]
    return contacts

x = sorted(glob('NTL9-0-protein-*.dcd'))

pool = multiprocessing.Pool(32)

for i in np.arange(0,len(x),32):
    contacts = pool.map(calc_contacts, x[i:i+32])
    for index,j in enumerate(range(i, i+32)):
        if index < 10:
            np.save('NTL9-0-protein-00{}.npy'.format(j), contacts[index])
        elif index < 100:
            np.save('NTL9-0-protein-0{}.npy'.format(j), contacts[index])
        else:
            np.save('NTL9-0-protein-{}.npy'.format(j), contacts[index])    

    
