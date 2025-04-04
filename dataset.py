import torch
import torch.nn as nn
import torchaudio
from torchaudio import transforms
from torch.utils.data import Dataset, DataLoader

import pandas as pd

class TriggerWordDataset(Dataset):
    def __init__(self, annotations_file : str):
        super().__init__()

        self.annotations_file = annotations_file
        self.df = pd.read_csv(self.annotations_file)
        self.labels = ['positive', 'negative', 'background']
        self.target_length = 500000
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, id):
        item = self.df.iloc[id]
        
        path = item["path"]

        waveform, sr = torchaudio.load(path, normalize=True) # channel=0  tells the function to load the audio in only mono channel
        if waveform.ndim > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = self._trim_or_pad_waveform(waveform)


        resr = 44100
        transform = nn.Sequential(
            # transforms.Resample(
            #     orig_freq=sr,
            #     new_freq=resr
            # ),
            transforms.MFCC(
                sample_rate=resr,
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

def create_dataset(ds):
    train_data, test_data = torch.utils.data.random_split(ds, [.8, .2])

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

if __name__ == "__main__":
    ds = TriggerWordDataset("./annotations_file.csv")

    train_data, test_data = create_dataset(ds)
    train_dataloader, test_dataloader = create_dataloaders(train_data, test_data, 16)
    print(next(iter(train_dataloader))[0].shape)