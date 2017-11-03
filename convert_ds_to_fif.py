# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 15:59:45 2017

@author: mje
"""

import glob
import mne

files = glob.glob("*.ds")
files.sort()

subjects = glob.glob("*_*")
subjects.sort()
subjects.remove(subjects[-1])

for subject in subjects[-2:]:
    files = glob.glob(subject + "/*.ds")

    raws = []
    for i, file in enumerate(files):
        raws.append(mne.io.read_raw_ctf(file))

    raw = mne.concatenate_raws(raws)
    raw.save(subject + "-raw.fif")
