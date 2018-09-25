import numpy as np
import matplotlib.pyplot as plt
import os
from mne.stats import fdr_correction
from scipy import stats

os.chdir('/Users/au194693/tmp/data/ks_decode')

subjects = ['as', 'ac', 'am', 'bk', 'gg', 'js', 'ks', 'ss']
conditions = ['all', 'left', 'right']
times = np.linspace(-0.1, 1.435, 308)

for condition in conditions:
    for subject in subjects:
        if condition is 'all':
            scores = np.load('%s_time_gen_score.npy' % subject)
        else:
            scores = np.load('%s_time_gen_score_%s.npy' % (subject, condition))

        scores = np.mean(scores, axis=0)

        # Plot the diagonal (it's exactly the same as the time-by-time decoding above)
        fig, ax = plt.subplots()
        ax.plot(times, np.diag(scores), label='score')
        ax.axhline(.5, color='k', linestyle='--', label='chance')
        ax.set_xlabel('Times')
        ax.set_ylabel('AUC')
        ax.legend()
        ax.axvline(.0, color='k', linestyle='-')
        ax.set_title('Decoding MEG sensors over time')
        plt.savefig('%s_%s_diag.png' % (subject, condition))

        # Plot the full matrix
        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(
            scores,
            interpolation='lanczos',
            origin='lower',
            cmap='RdBu_r',
            extent=times[[0, -1, 0, -1]],
            vmin=0.,
            vmax=1.)
        ax.set_xlabel('Testing Time (s)')
        ax.set_ylabel('Training Time (s)')
        ax.set_title('Temporal Generalization')
        ax.axvline(0, color='k')
        ax.axhline(0, color='k')
        plt.colorbar(im, ax=ax)
        plt.savefig('%s_%s_TG.png' % (subject, condition))

for condition in conditions:
    scores_all = np.empty((len(subjects), 308, 308))

    for jj, subject in enumerate(subjects):
        if condition is 'all':
            scores_all[jj] = np.load(
                '%s_time_gen_score.npy' % subject).mean(axis=0)
        else:
            scores_all[jj] = np.load(
                '%s_time_gen_score_%s.npy' % (subject, condition)).mean(axis=0)

    tstats_diag, pvals_diag = stats.ttest_1samp(np.diag(scores_all), 0.5)
    tstat_tg, pvals_tg = stats.ttest_1samp(scores_all, 0.5)
    rejectd_tg, pvals_tg_fdr = fdr_correction(pvals_tg)
    rejectd_diag, pvals_diag_fdr = fdr_correction(np.diag(pvals_tg))
    threshold_fdr = np.min(np.abs(np.diag(tstat_tg))[rejectd_diag])

    scores_all = scores_all.mean(axis=0)
    # Plot the diagonal (it's exactly the same as the time-by-time decoding above)
    fig, ax = plt.subplots()
    ax.plot(times, np.diag(scores_all.mean(axis=0)), label='score')
    ax.axhline(.5, color='k', linestyle='--', label='chance')
    ax.set_xlabel('Times')
    ax.set_ylabel('AUC')
    ax.legend()
    ax.axvline(.0, color='k', linestyle='-')
    ax.set_title('Decoding MEG sensors over time')
    plt.savefig('all_%s_diag.png' % condition)

    fig, ax = plt.subplots()
    ax.plot(times, np.diag(tstat_tg), label='score')
    ax.axhline(.5, color='k', linestyle='--', label='chance')
    ax.axhline(threshold_fdr, color='k',  linestyle='--', label='fdr threshold')
    ax.set_xlabel('Times')
    ax.set_ylabel('AUC')
    ax.legend()
    ax.axvline(.0, color='k', linestyle='-')
    ax.set_title('Decoding MEG sensors over time')
    plt.show()
    # plt.savefig('all_%s_diag.png' % condition)

    # Plot the full matrix
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
        scores_all.mean(axis=0),
        interpolation='lanczos',
        origin='lower',
        cmap='RdBu_r',
        extent=times[[0, -1, 0, -1]],
        vmin=0.,
        vmax=1.)
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Temporal Generalization')
    ax.axvline(0, color='k')
    ax.axhline(0, color='k')
    plt.colorbar(im, ax=ax)
    plt.savefig('all_%s_TG.png' % condition)
