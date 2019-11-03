# TODO: include options for which panels to make log-scale or linear scale...
# TODO: include real empirical estimates of the MFPT for comparison
# TODO: label the threshold that was used to guarantee that there are 10 intermediate microstates (JDC: Perhaps we could show those as dotted or red lines or something, just so it's visually obvious.)
# TODO: use "observation interval" \Delta t instead of "lag-time" \tau
# TODO: go out to potentially much longer lag-times? 10^5 ns?
# TODO: include and propagate uncertainty estimates from the empirical ACF in the implied rate estimates?

import numpy as np
from simtk import unit
import matplotlib.pyplot as plt


from seaborn.apionly import color_palette
thresholds = [0.5, 0.8, 0.9, 0.95, 0.99, 0.999]
colors = color_palette("GnBu_d", n_colors=len(thresholds))

dt = 0.0002 * unit.microseconds
max_lag_time = 5000 * unit.nanoseconds
max_lag_ind = int(max_lag_time / dt)

print('plotting out to a max observation interval (aka lag-time) of ', max_lag_time)

# MEMBERSHIPS
chignolin_membs = np.load('chignolin_lag150ns_metastable_memberships.npy')

villin_membs = np.load('villin_lag100ns_metastable_memberships_3states.npy')
villin_misfold = np.load('villin_misfold.npy')

trpcage_membs = np.load('trpcage_lag100ns_metastable_memberships.npy')

ntl9_membs = np.load('ntl9_lag200ns_metastable_memberships.npy')

from statsmodels.tsa.stattools import acf
def compute_implied_rates(h_A, h_B, tmax=100, alpha=0.95):
    """based on """
    T = len(h_A)
    if tmax >= T:
        tmax = T - 1

    # compute stationary probability
    PA = np.mean(h_A)
    PB = np.mean(h_B)

    # compute normalized time-correlation function for dh_A
    Cdt = np.ones(tmax + 1)
    dh_A = h_A - PA
    denom = np.mean(dh_A ** 2)
    for t in range(1, tmax):
        Cdt[t] = np.mean(dh_A[:T - t] * dh_A[t:]) / denom
    tval = np.arange(1, tmax + 1)

    # implied rate constant k at time t
    kim_t = - np.log(Cdt[1:]) / tval
    return kim_t


from tqdm import tqdm



def get_rate_estimates_at_membership_thresholds(dtraj, A_membs, thresholds, dt=0.2, tmax=1000):
    tval = np.arange(1, tmax + 1) * dt

    Cdts = []
    kim_ts = []

    for thresh in tqdm(thresholds):
        A_inds = np.arange(len(A_membs))[A_membs >= thresh]
        h_A = sum([dtraj == i for i in A_inds]) > 0
        PA = np.mean(h_A)
        PB = 1.0 - PA

        Cdt = acf(h_A, nlags=tmax, fft=True)
        Cdts.append(Cdt)
        kim_t = - PB * np.log(Cdt[1:]) / tval
        kim_ts.append(kim_t)
    return thresholds, tval, Cdts, kim_ts


def remove_top_right_spines(ax):
    """Aesthetic tweak of matplotlib axes"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_ACF(ax, tval, Cdt):
    pass

def plot_rates(ax, tval, kim_t):
    pass

def plot_MFPT(ax, tval, kim_t):
    pass

thresholds, tval, Cdts, kim_ts = get_rate_estimates_at_membership_thresholds(
    dtraj=trpcage_dtrajs[0],
    A_membs=trpcage_membs[:, 0],
    thresholds=thresholds,
    dt=dt / unit.nanoseconds,
    tmax=max_lag_ind,
)

thresholds_rev, tval_rev, Cdts_rev, kim_ts_rev = get_rate_estimates_at_membership_thresholds(
    dtraj=trpcage_dtrajs[0],
    A_membs=trpcage_membs[:, 0],
    thresholds=thresholds,
    dt=dt / unit.nanoseconds,
    tmax=max_lag_ind,
)

def plot(thresholds, tval, Cdts, kim_ts, Cdts_rev, kim_ts_rev):
    """"""


    ## ACFs
    # ACF A->B
    ax1 = plt.subplot(3, 2, 1)
    remove_top_right_spines(ax1)
    for i, Cdt in enumerate(Cdts):
        plt.plot(tval, Cdt[1:], label=thresholds[i], c=colors[i])
    # plt.xlabel(r'lag-time $\tau$ (ns)')
    # plt.ylim(0,1)
    plt.ylabel('autocorrelation function')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.title(r'Trp cage $A \to B$')
    plt.legend(title='membership threshold', loc='best')

    # ACF B->A
    ax2 = plt.subplot(3, 2, 2)
    remove_top_right_spines(ax2)
    for i, Cdt in enumerate(Cdts_rev):
        plt.plot(tval, Cdt[1:], label=thresholds[i], c=colors[i])
    # plt.xlabel(r'lag-time $\tau$ (ns)')
    # plt.ylim(0,1)
    # plt.ylabel('implied rate estimate (1/ns)')
    plt.title(r'Trp cage $B \to A$')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.legend(title='membership threshold')

    ## RATES
    # rate A->B
    ax1 = plt.subplot(3, 2, 3)
    remove_top_right_spines(ax1)
    for i, kim_t in enumerate(kim_ts):
        plt.plot(tval, kim_t, label=thresholds[i], c=colors[i])
    # plt.xlabel(r'lag-time $\tau$ (ns)')
    y_rate_min = kim_t[-1]
    y_rate_max = kim_t[1000]
    # plt.ylim(y_rate_min,y_rate_max)
    plt.ylabel('implied rate estimate (1/ns)')
    # plt.xscale('log')
    plt.yscale('log')
    # plt.title(r'chignolin $A \to B$')
    # plt.legend(title='membership threshold')

    # rate B->A
    ax2 = plt.subplot(3, 2, 4)
    remove_top_right_spines(ax2)
    for i, kim_t in enumerate(kim_ts_rev):
        plt.plot(tval_rev, kim_t, label=thresholds[i], c=colors[i])
    # plt.xlabel(r'lag-time $\tau$ (ns)')
    # plt.ylim(y_rate_min,y_rate_max)
    # plt.ylabel('implied rate estimate (1/ns)')
    # plt.xscale('log')
    plt.yscale('log')
    # plt.title(r'chignolin $B \to A$')
    # plt.legend(title='membership threshold')

    ## MFPTs
    # MFPT A->B
    ax3 = plt.subplot(3, 2, 5)
    remove_top_right_spines(ax3)

    for i, kim_t in enumerate(kim_ts):
        plt.plot(tval, 1 / kim_t, label=thresholds[i], c=colors[i])
    plt.xlabel(r'lag-time $\tau$ (ns)')
    plt.ylabel('MFPT estimate (ns)')
    # plt.xscale('log')
    plt.yscale('log')
    # plt.legend(title='membership threshold')

    # MFPT B->A
    ax4 = plt.subplot(3, 2, 6)
    remove_top_right_spines(ax4)

    for i, kim_t in enumerate(kim_ts_rev):
        plt.plot(tval_rev, 1 / kim_t, label=thresholds[i], c=colors[i])
    plt.xlabel(r'lag-time $\tau$ (ns)')
    # plt.xscale('log')
    plt.yscale('log')
    # plt.ylabel('MFPT estimate (ns)')
    # plt.legend(title='membership threshold', loc=(1,0))
    plt.tight_layout()


plt.figure(figsize=(7, 7))
plot_trpcage(thresholds)
plt.savefig('trpcage_acf_based_rate_estimates.pdf', bbox_inches='tight')
plt.savefig('trpcage_acf_based_rate_estimates.png', dpi=300, bbox_inches='tight')