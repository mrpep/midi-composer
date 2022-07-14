import pandas as pd
from pathlib import Path
from midi_extractors import midi_to_events
from torch.utils.data import Dataset
import random
import numpy as np

def load_maestro(dataset_path='../datasets/maestro-v3.0.0', cache_path='maestro_dataset.pkl'):
    if Path(cache_path).exists():
        maestro_metadata = pd.read_pickle(cache_path)
    else:
        maestro_metadata = pd.read_csv(Path(dataset_path,'maestro-v3.0.0.csv'))
        maestro_metadata['seq_in'] = maestro_metadata['midi_filename'].progress_apply(lambda x: midi_to_events(Path(dataset_path,x)))
        maestro_metadata.to_pickle(cache_path)
    return maestro_metadata

class MaestroDataset(Dataset):
    def __init__(self, df, max_seq_len=200):
        self.data = df
        self.max_seq_len = max_seq_len
        self.i_to_idx = {}
        counter_i = 0
        for idx, (row_idx, row) in enumerate(self.data.iterrows()):
            n_idxs = int(row['duration'])
            dict_i = {k: idx for k in range(counter_i,counter_i+n_idxs)}
            counter_i += n_idxs
            self.i_to_idx.update(dict_i)
            
    def __getitem__(self,i):
        idx = self.i_to_idx[i]
        seq_in = self.data.iloc[idx]['seq_in']
        start_idx = random.randint(0,len(seq_in)-self.max_seq_len)
        out = np.array(seq_in[start_idx:start_idx+self.max_seq_len])
        out[out>355]=355
        return out
    
    def __len__(self):
        return len(self.i_to_idx)
    
