from random import sample
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence
import os
import pandas as pd
from pydub import AudioSegment
# First is blank char
CHARS = "* abcdefghijklmnopqrstuvwxyzåäö'.,"

class Collate:
    def __call__(self, batch):
        spectrograms = []
        labels = []
        spectrogram_lengths = []
        label_lengths = []

        for (spectrogram, label, spectrogram_length, label_length) in batch:
            if spectrogram is None:
                continue
        # print(spectrogram.shape)
            spectrograms.append(spectrogram.squeeze(0).transpose(0, 1))
            labels.append(label)
            spectrogram_lengths.append(spectrogram_length)
            # spectrogram_lengths.append(0) # infers length instead?
            label_lengths.append(label_length)
            # label_lengths.append(0)

        spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
        
        return spectrograms, labels, spectrogram_lengths, label_lengths


def get_feature(waveform, sample_rate, n_mels):
    hop_length = int(sample_rate/100)
    win_length = int(sample_rate/40)
    spectrogram: torch.Tensor = torchaudio.transforms.MelSpectrogram(
        sample_rate = sample_rate,
        n_mels = n_mels,
        n_fft = win_length,
        hop_length = hop_length,
        win_length = win_length
    )(waveform)
    spectrogram = torch.log(spectrogram + 1e-14)
    return spectrogram

class CommonVoiceDataset(Dataset):


    def __init__(self, path:os.PathLike, tsv_file:str, sep=',', n_mels=50, reduce_factor=1):
        # reduce factor = amount of stride/pooling etc that reduces input size
        self.clip_path = os.path.join(path, 'clips')
        self.meta_path = os.path.join(path, tsv_file)
        self.chars = CHARS
        self.meta = pd.read_csv(self.meta_path, sep=sep)
        self.n_mels = n_mels
        self.reduce_factor = reduce_factor
    
    def filter(self, x):
        return ''.join(c for c in x if c in self.chars)

    def translate_str_to_list_of_int(self, s):
        """Converts string to list of integers"""
        return [self.chars.index(c) for c in s]

    def translate_from_list_of_int(self, l):
        """Converts list of indices to list of chars"""
        res = ""
        for index in l:
            res += self.chars[index]
        
        return res

    def translate_from_logits(self, preds:torch.Tensor):
        # Input shape = (Batch Size, Length, Chars)
        predicted_sentences = []
        for i in range(preds.size(0)):
            predicted_indices = torch.argmax(torch.log_softmax(preds[i], -1), -1).numpy()

            predicted_sentences.append(self.translate_from_list_of_int(predicted_indices))

        return predicted_sentences

    def __len__(self):
        return len(self.meta)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load the file
        clip_path = os.path.join(self.clip_path, self.meta.path[idx])
        waveform, sample_rate = torchaudio.load(clip_path)

        spectrogram = get_feature(waveform, sample_rate, self.n_mels)
        label = self.meta.sentence[idx].lower()
        label = self.translate_str_to_list_of_int(self.filter(label))
        if torch.any(torch.isnan(spectrogram)):
            print("NAN DETECTED")
        return (
            spectrogram, 
            torch.tensor(label, dtype=torch.long), 
            spectrogram.size(-1)//self.reduce_factor, 
            len(label)//self.reduce_factor
        )




if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = CommonVoiceDataset('data', 'validated.csv')

    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=Collate())

    next(iter(train_loader))

    print(len(dataset.chars))