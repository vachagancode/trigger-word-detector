import torch
import torchaudio
import random

class AudioAugmenter:
    def __init__(self, sample_rate=44100, eps : float = 1e-5):
        self.sample_rate = sample_rate
        self.eps = eps

    def time_shift(self, audio, shift_limit : float = 0.2):
        """
            Shifts audio randomly in time
        """

        shift = int(random.random() * shift_limit * audio.shape[1] + self.eps) 
        direction = random.choice([-1, 1])
        shift = shift * direction

        if shift > 0:
            # Shift to the right
            padded = torch.zeros_like(audio)
            padded[:, shift:] = audio[:, :audio.shape[1] - shift]
            return padded
        else:
            # Shift to the left
            shift = -shift
            padded = torch.zeros_like(audio)
            padded[:, :audio.shape[1]-shift] = audio[:, shift:]
            return padded
    
    def add_noise(self, audio, noise_level : float = 0.005):
        """
            Adds noise to the audio
        """

        signal_magnitude = torch.mean(torch.abs(audio)) if torch.mean(torch.abs(audio)) > 0 else 1e-6 # TODO: Understand what's going on here
        noise_level *= signal_magnitude
        noise = torch.randn_like(audio) * noise_level
        return audio + noise

    def apply_random_augmentations(self, audio, prob=0.45):
        augmented = audio.clone()

        if random.random() < prob:
            augmented = self.time_shift(augmented)
        if random.random() < prob:
            augmented = self.add_noise(augmented)

        return augmented


if __name__ == "__main__":
    
    waveform, sr = torchaudio.load("./custom_data/positive/Arman.mp3", normalize=True)
    ag =  AudioAugmenter(sample_rate=sr)

    if waveform.ndim > 1 and waveform.shape[0] > 1: # the first dimension show how many channels do we have
        waveform = torch.mean(waveform, dim=0)
        waveform = waveform.unsqueeze(dim=0)
    