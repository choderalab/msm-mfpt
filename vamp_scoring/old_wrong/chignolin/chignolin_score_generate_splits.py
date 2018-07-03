import sklearn
import sklearn.model_selection

traj_files = glob("*.dcd")

shuffle_split = sklearn.model_selection.ShuffleSplit(n_splits=10, test_size=0.5)
splits = np.array(list(shuffle_split.split(traj_files)))
np.save('chignolin_vamp1_splits', splits)
