import torch
import torch.nn as nn
import torchaudio
from torchaudio import transforms
from torch.utils.data import Dataset, DataLoader

from audio_augmenter import AudioAugmenter

import pandas as pd

class TriggerWordDataset(Dataset):
    def __init__(self, annotations_file : str, train : bool = False):
        super().__init__()
        self.train = train
        self.annotations_file = annotations_file
        self.df = pd.read_csv(self.annotations_file)
        self.ddf = self.df[:int(len(self.df)*0.8)] if self.train else self.df[int(len(self.df)*0.2):]
        self.labels = ['positive', 'negative', 'background']
        self.target_length = 500000
        self.resr = 22050
        self.agm = AudioAugmenter(sample_rate=self.resr)
        
    def __len__(self):
        return len(self.ddf)

    def __getitem__(self, id):
        item = self.ddf.iloc[id]
        
        path = item["path"]

        waveform, sr = torchaudio.load(path, normalize=True) 
        if waveform.ndim > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = self._trim_or_pad_waveform(waveform)
        
        transform = nn.Sequential(
            # transforms.Resample(
            #     orig_freq=sr,
            #     new_freq=resr
            # ),
            transforms.MFCC(
                sample_rate=self.resr,
                n_mfcc=13,
                melkwargs={'n_fft': 400, "hop_length" : 512}
            ),
            # transforms.FrequencyMasking(freq_mask_param=20),
            # transforms.TimeMasking(time_mask_param=35)
        )
        mfcc = transform(waveform) 
        deltas = torchaudio.functional.compute_deltas(mfcc)
        delta_deltas = torchaudio.functional.compute_deltas(deltas)

        output = torch.cat([mfcc, deltas, delta_deltas], dim=1)

        output = output.squeeze(0)

        label = self.labels.index(item["label"])

        return output, label, sr


    def _trim_or_pad_waveform(self, waveform):
        if waveform.shape[1] < self.target_length:
            # pad zeros to the end of the waveform
            zeros = torch.zeros((1, self.target_length - waveform.shape[1]), dtype=torch.float32)

            waveform = torch.cat((waveform, zeros), dim=1)
        else:
            waveform = waveform[:, :self.target_length]

        return waveform 

def create_dataset():
    train_data = TriggerWordDataset("./annotations_file.csv", train=True)
    test_data = TriggerWordDataset("./annotations_file.csv", train=False)

    return train_data, test_data

def create_dataloaders(train_data, test_data, batch_size : int = 8):
    train_dataloader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_data,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True
    )

    return train_dataloader, test_dataloader
