import os
import torch
import torch.nn as nn
import torchaudio
from torchaudio import transforms

import math
import matplotlib.pyplot as plt
import librosa.display
from random import randint
from tqdm import tqdm

from dataset import TriggerWordDataset, create_dataset, create_dataloaders
from model import create_model
from config import get_config

def plot_random_spectrogram(dataset):
    spectrogram, label, sr = dataset[randint(0, len(dataset))]
    plt.figure(figsize=(7, 10))
    librosa.display.specshow(
        data=spectrogram.squeeze().numpy(),
        sr=sr,
        x_axis="time",
        y_axis="log"
    )
    plt.colorbar(format="%+2.f dB")
    plt.plot()
    plt.show()

def train(
    cfg, dataset, m=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup the model
    model = create_model(
        in_features=cfg["in_features"],
        out_features=cfg["out_features"],
        hidden_layers=cfg["hidden_layers"],
        device=device
    )

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00009)

    # data setup
    train_data, test_data = create_dataset(dataset)
    train_dataloader, test_dataloader = create_dataloaders(train_data, test_data)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.00015, pct_start=0.3, total_steps=cfg["epochs"]*int(len(train_dataloader)))

    if m is not None:
        data = torch.load(f=m, map_location=device)

        model_state_dict = data["model_state_dict"]
        optimizer_state_dict = data["optimizer_state_dict"]
        scheduler_state_dict = data["scheduler_state_dict"]

        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        scheduler_state_dict["total_steps"] = scheduler_state_dict["total_steps"] + cfg["epochs"]*int(len(train_dataloader))
        scheduler.load_state_dict(scheduler_state_dict)
        start_epoch = data["epoch"]
    else:
        start_epoch = 0

    print(f"Using: {device}\nEpochs to train: {cfg['epochs']}")

    previous_test_loss = float('inf')
    for epoch in range(start_epoch, start_epoch + cfg["epochs"]+1):
        batch_loader = tqdm(train_dataloader)
        test_loss, train_loss = 0, 0
        step = 0
        for X, y, _ in batch_loader:
            model.train()

            X, y = X.to(device), y.to(device)

            # forward pass 
            y_logits = model(X)
            # calculate the loss 
            loss = loss_fn(y_logits, y)
            train_loss += loss
            step += 1

            # optimizer zero grad
            optimizer.zero_grad()

            # loss backward
            loss.backward()

            # optimizer step
            optimizer.step()
            scheduler.step()

        test_step = 0
        if epoch % 2 == 0:
            model.eval()
            with torch.inference_mode():
                test_batch_loader = (test_dataloader)
                for X_test, y_test, _ in test_batch_loader:
                    X_test, y_test = X_test.to(device), y_test.to(device)

                    # forward pass 
                    y_test_logits = model(X_test)
                    # calculate the loss 
                    loss_test = loss_fn(y_test_logits, y_test)
                    test_loss += loss_test
                    test_step += 1

                train_loss /= step
                test_loss /= test_step
            print(f"Epoch: {epoch} | Train Loss: {train_loss:.2f} | Test Loss: {test_loss:.2f}")
            
            # Save the model
            previous_test_loss = test_loss
            model_data = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "lr": scheduler.get_last_lr()
            }

            torch.save(
                obj=model_data,
                f=f"./models/me{epoch}l{math.floor(test_loss*100)}.pth"
            )
    return model

def audio_to_mfcc(ds, path : str):
    waveform, sr = torchaudio.load(path, normalize=True)
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    resr = 44100
    waveform = ds._trim_or_pad_waveform(waveform)
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
    )
    mfcc = transform(waveform) 
    deltas = torchaudio.functional.compute_deltas(mfcc)
    delta_deltas = torchaudio.functional.compute_deltas(deltas)

    output = torch.cat([mfcc, deltas, delta_deltas], dim=1)

    return output

def classify(ds, model : str, path : str, cfg : str):
    # labels
    labels = ['positive', 'negative', 'background']
    # labels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_data = torch.load(f=model, map_location=device)

    model = create_model(
        in_features=cfg["in_features"],
        out_features=cfg["out_features"],
        hidden_layers=cfg["hidden_layers"],
        device=device
    )
    model.load_state_dict(model_data["model_state_dict"])

    mel_spectrogram = audio_to_mfcc(ds, path)

    # Do the forward pass
    y_logits = model(mel_spectrogram.unsqueeze(0))

    y_preds = torch.softmax(y_logits, dim=1)
    prediction = torch.argmax(y_preds, dim=1)
    print(y_preds)

    return labels[prediction]

    # return f"`{labels[prediction]}` with the probability of {(prob * 100):.2f} %"

def rename_audio_files(dir : str):
    """
    Simply numerates files in the specified directory the following way - "1.mp3", "2.mp3", etc.
    This return files in the root directory, then replace them manually
    """
    directory_list = os.listdir(dir)
    for idx, file in enumerate(directory_list):
        os.rename(f"{dir}/{file}", f"{idx}.mp3")

def reform_model(path : str, device, model_name):
    """
    This function simply removes all the unnecessary information from the model and keeps only its state_dict  
    """
    model = create_model(device=device)
    data = torch.load(f=path, map_location=device)
    
    torch.save(
        obj=data["model_state_dict"],
        f=f"./valid_models/{model_name}.pth"
    )


if __name__ == "__main__":
    ds = TriggerWordDataset("./annotations_file.csv")
    cfg = get_config()
    # train(dataset=ds, cfg=cfg)

    print(classify(ds, "./models/me178l25.pth", "./custom_data/positive/Arman.mp3", cfg))

    # rename_audio_files(dir="./raw_data/negative")