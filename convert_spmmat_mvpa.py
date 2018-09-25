import os

import mne
import numpy as np
import scipy.io as sio
from mne.datasets import spm_face
from mne.decoding import GeneralizingEstimator, cross_val_multiscore
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

os.chdir('/Users/au194693/tmp/data/ks_decode')

n_jobs = 3
subjects = ['as', 'ac', 'am', 'bk', 'gg', 'js', 'ks', 'ss']

events_id = {
    # "onset": 1,
    "left/grating": 2,
    "left/face": 3,
    "right/face": 4,
    "right/grating": 5
}

# Data information
data_path = spm_face.data_path()
raw_fname = data_path + '/MEG/spm/SPM_CTF_MEG_example_faces%d_3D.ds'
raw = mne.io.read_raw_ctf(raw_fname % 1, preload=True)  # Take first run
raw.pick_types(meg='mag', ref_meg=False)
raw.info['sfreq'] = 200

# loop over subjects, first conver the data, the extract and run MVPA
for subject in subjects:
    D = sio.loadmat('%s_data.mat' % subject)
    trials = sio.loadmat('%s_trials.mat' % subject)

    data = D['dat']
    trials = trials['trials']

    info = mne.create_info(raw.ch_names, sfreq=200, ch_types='mag')
    epochs = mne.EpochsArray(data, info, tmin=-0.1)
    epochs.events[:, 2] = trials.reshape(-1)
    epochs.event_id = events_id
    epochs.save('%s-epo.fif' % subject)

    epochs.equalize_event_counts(epochs.event_id)

    X = np.concatenate((epochs["face"].get_data(),
                        epochs["grating"].get_data()))
    y = np.concatenate((np.zeros(len(epochs["face"].get_data())),
                        np.ones(len(epochs["grating"].get_data()))))
    cv = StratifiedKFold(n_splits=10, shuffle=True)

    clf = make_pipeline(StandardScaler(), LogisticRegression())
    time_gen = GeneralizingEstimator(clf, scoring='roc_auc', n_jobs=n_jobs)
    time_gen.fit(X, y)
    scores = cross_val_multiscore(time_gen, X, y, cv=cv)

    # Save results
    joblib.dump(time_gen, "%s_time_gen.jbl" % subject)
    np.save("%s_time_gen_score.npy" % subject, scores)

    X_left = np.concatenate((epochs["left/face"].get_data(),
                             epochs["left/grating"].get_data()))
    y_left = np.concatenate((np.zeros(len(epochs["left/face"].get_data())),
                             np.ones(len(epochs["left/grating"].get_data()))))
    clf = make_pipeline(StandardScaler(), LogisticRegression())
    time_gen_left = GeneralizingEstimator(
        clf, scoring='roc_auc', n_jobs=n_jobs)
    time_gen_left.fit(X_left, y_left)
    scores_left = cross_val_multiscore(time_gen, X_left, y_left, cv=cv)

    # Save results
    joblib.dump(time_gen_left, "%s_time_gen_left.jbl" % subject)
    np.save("%s_time_gen_score_left.npy" % subject, scores_left)

    X_right = np.concatenate((epochs["right/face"].get_data(),
                              epochs["right/grating"].get_data()))
    y_right = np.concatenate(
        (np.zeros(len(epochs["right/face"].get_data())),
         np.ones(len(epochs["right/grating"].get_data()))))
    clf = make_pipeline(StandardScaler(), LogisticRegression())
    time_gen_right = GeneralizingEstimator(
        clf, scoring='roc_auc', n_jobs=n_jobs)
    time_gen_right.fit(X_right, y_right)
    scores_right = cross_val_multiscore(time_gen, X_right, y_right, cv=cv)

    # Save results
    joblib.dump(time_gen_right, "%s_time_gen_right.jbl" % subject)
    np.save("%s_time_gen_score_right.npy" % subject, scores_right)
