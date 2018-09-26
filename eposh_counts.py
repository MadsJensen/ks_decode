import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir('/Users/au194693/tmp/data/ks_decode')


subjects = ['as', 'ac', 'am', 'bk', 'gg', 'js', 'ks', 'ss']

count_df = pd.DataFrame()

for subject in subjects:
    epochs_counts = np.load('%s_epochs_range.npy' % subject)
    row = pd.DataFrame([{
        'subject': subject,
        'epoch_count': epochs_counts.shape[0],
        'face_percent': sum((epochs_counts < int(len(epochs_counts) / 2))) /
                        len(epochs_counts) * 100
    }])
    count_df = pd.concat((count_df, row), ignore_index=True)

count_df = count_df[['subject', 'epoch_count', 'face_percent']]
count_df.to_csv('epochs_counts.csv', index=False)

# plot epochs
x = np.arange(len(count_df.subject.values))
sns.barplot(x='subject', y='epoch_count', data=count_df, color='k')
plt.savefig('plots/epochs_count.png')