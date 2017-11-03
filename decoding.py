"""
Script for applying 'Generalization across time'.

@author: mje
@email: mje.mads [] gmail com

"""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import mne
from mne.decoding import GeneralizingEstimator, cross_val_multiscore
from glob import glob
import numpy as np

from my_settings import epochs_folder, mvpa_folder

epochs_list = glob(epochs_folder + "*-epo.fif")
epochs_list.sort()

for e in epochs_list:
    subj_num = e.split("/")[-1].split("_")[0]
    epochs = mne.read_epochs(e)
    epochs.equalize_event_counts(epochs.event_id)

    X = np.concatenate((epochs["face"].get_data(),
                        epochs["grating"].get_data()))
    y = np.concatenate((np.zeros(len(epochs["face"].get_data())), np.ones(
        len(epochs["grating"].get_data()))))
    cv = StratifiedKFold(n_splits=10, shuffle=True)

    clf = make_pipeline(StandardScaler(), LogisticRegression())
    time_gen = GeneralizingEstimator(clf, scoring='roc_auc', n_jobs=1)
    time_gen.fit(X, y)
    scores = cross_val_multiscore(time_gen, X, y, cv=cv)

    # Save results
    joblib.dump(time_gen, mvpa_folder + "%s_time_gen.jbl" % subj_num)
    np.save(mvpa_folder + "%s_time_gen_score.npy" % subj_num, scores)

    X_left = np.concatenate((epochs["left/face"].get_data(),
                             epochs["left/grating"].get_data()))
    y_left = np.concatenate((np.zeros(len(epochs["left/face"].get_data())),
                             np.ones(len(epochs["left/grating"].get_data()))))
    clf = make_pipeline(StandardScaler(), LogisticRegression())
    time_gen_left = GeneralizingEstimator(clf, scoring='roc_auc', n_jobs=1)
    time_gen_left.fit(X_left, y_left)
    scores_left = cross_val_multiscore(time_gen, X_left, y_left, cv=cv)

    # Save results
    joblib.dump(time_gen_left, mvpa_folder + "%s_time_gen_left.jbl" % subj_num)
    np.save(mvpa_folder + "%s_time_gen_score_left.npy" % subj_num, scores_left)

    X_right = np.concatenate((epochs["right/face"].get_data(),
                              epochs["right/grating"].get_data()))
    y_right = np.concatenate(
        (np.zeros(len(epochs["right/face"].get_data())), np.ones(
            len(epochs["right/grating"].get_data()))))
    clf = make_pipeline(StandardScaler(), LogisticRegression())
    time_gen_right = GeneralizingEstimator(clf, scoring='roc_auc', n_jobs=1)
    time_gen_right.fit(X_right, y_right)
    scores_right = cross_val_multiscore(time_gen, X_right, y_right, cv=cv)

    # Save results
    joblib.dump(time_gen_right,
                mvpa_folder + "%s_time_gen_right.jbl" % subj_num)
    np.save(mvpa_folder + "%s_time_gen_score_right.npy" % subj_num,
            scores_right)
