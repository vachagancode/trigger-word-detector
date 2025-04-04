import torch
import torch.nn as nn

from dataset import TriggerWordDataset

class TriggerWordDetector(nn.Module):
    def __init__(self, in_features : int = 1, out_features : int = 3, hidden_layers : int= 32, dropout : float = 0.35): # we expect 3 different predictions - 1.positive, 2.negative, 3.background
        super().__init__()
        self.layer_1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_features,
                    out_channels=hidden_layers,
                    kernel_size=3
                ),
                nn.BatchNorm2d(hidden_layers),
                nn.ReLU(),
                nn.MaxPool2d(
                    kernel_size=2,
                    stride=1
                ),
                nn.Conv2d(
                    in_channels=hidden_layers,
                    out_channels=hidden_layers,
                    kernel_size=3
                ),
                nn.BatchNorm2d(hidden_layers),
                nn.ReLU(),
        )
        self.layer_2 = nn.Sequential(
                nn.Conv2d(
                    in_channels=hidden_layers,
                    out_channels=hidden_layers,
                    kernel_size=3
                ),
                nn.BatchNorm2d(hidden_layers),
                nn.ReLU(),
                nn.MaxPool2d(
                    kernel_size=2,
                    stride=1
                ),
                nn.BatchNorm2d(hidden_layers),
                nn.Conv2d(
                    in_channels=hidden_layers,
                    out_channels=hidden_layers,
                    kernel_size=3
                ),
                nn.ReLU(),
        )
        self.layer_3 = nn.Sequential(
                nn.Conv2d(
                    in_channels=hidden_layers,
                    out_channels=hidden_layers,
                    kernel_size=1
                ),
                nn.BatchNorm2d(hidden_layers),
                nn.ReLU(),
                nn.MaxPool2d(
                    kernel_size=1,
                    stride=1,
                ),
                nn.BatchNorm2d(hidden_layers),
                nn.Conv2d(
                    in_channels=hidden_layers,
                    out_channels=hidden_layers,
                    kernel_size=1
                ),
                nn.ReLU(),
        )

        self.flatten =  nn.Flatten()

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=1794752, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=8, out_features=3)
        )

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.flatten(x)
        x = self.linear_layers(x)

        return x


def create_model(device : torch.device, in_features : int = 1 , out_features : int = 3, hidden_layers :  int = 32):
    model = TriggerWordDetector(
        in_features=in_features,
        out_features=out_features,
        hidden_layers=hidden_layers
    )

    model.to(device)
    return model 


if __name__ == "__main__":
    ds = TriggerWordDataset("./annotations_file.csv")
    spectrogram, label, sr = ds[6]

    model = TriggerWordDetector()
    print(model(spectrogram.unsqueeze(0)))