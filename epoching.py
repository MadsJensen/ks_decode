"""
Script for epoching data.

@author: mje
@email: mje.mads [] gmail.com

"""
import mne
from glob import glob
import numpy as np
from my_settings import raw_path, epochs_folder

raw_files = glob(raw_path + "*bp-raw.fif")
raw_files.sort()

tmin, tmax = -0.1, 1.0
baseline = (None, 0.)
reject = dict(mag=4e-12)

event_id = {
    "left/face": 111,
    "left/mixed": 112,
    "left/grating": 113,
    "right/face": 121,
    "right/mixed": 122,
    "right/grating": 123
}

event_id_5 = {
    "left/face": 111,
    "left/grating": 113,
    "right/face": 121,
    "right/grating": 123
}


def correct_events(events, raw):
    """Take a events array and change the trigger codes.

    Returns
    =======
    events : array

    """
    events_new = events.copy()
    for j in range(len(events_new) - 1):
        if events_new[j, 1] == 2:
            k = 1
            found_event = False
            while found_event is False:
                if events_new[j + k, 1] == 191:
                    events_new[j, 2] = 111
                    found_event = True
                elif events_new[j + k, 1] == 223:
                    events_new[j, 2] = 112
                    found_event = True
                elif events_new[j + k, 1] == 239:
                    events_new[j, 2] = 113
                    found_event = True
                elif (events_new[j + k, 1] == 2) or (
                        events_new[j + k, 1] == 5):
                    found_event = True
                else:
                    k += 1

        elif events_new[j, 1] == 5:
            k = 1
            found_event = False
            while found_event is False:
                if events_new[j + k, 1] == 254:
                    events_new[j, 2] = 121
                    found_event = True
                elif events_new[j + k, 1] == 253:
                    events_new[j, 2] = 122
                    found_event = True
                elif events_new[j + k, 1] == 251:
                    events_new[j, 2] = 123
                    found_event = True
                elif (events_new[j + k, 1] == 2) or (
                        events_new[j + k, 1] == 5):
                    found_event = True
                else:
                    k += 1

    adjust_time_line_by = -33
    events_new[:, 0] = [
        x - np.round(adjust_time_line_by * 10**-3 * raw.info['sfreq'])
        for x in events[:, 0]
    ]

    return events_new


for r in raw_files:
    raw = mne.io.read_raw_fif(r)
    events = mne.find_events(raw)
    events_corrected = correct_events(events, raw)

    epochs = mne.Epochs(
        raw,
        events_corrected,
        event_id_5,
        tmin,
        tmax,
        baseline=baseline,
        reject=reject,
        preload=True)
    epochs.save(epochs_folder +
                "%s_bp-epo.fif" % r.split("/")[-1].split("_")[0])
