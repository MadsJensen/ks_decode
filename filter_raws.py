import mne

from glob import glob
from my_settings import raw_path

files =  glob(raw_path + "*-raw.fif")
files.sort()

for file in files:
    raw = mne.io.read_raw_fif(file, preload=True)
    raw.filter(l_freq = 0.5, h_freq=None)
    raw.filter(l_freq = None, h_freq=40)
    raw.resample(200)
    raw.save(file.replace("-", "_bp-"))
