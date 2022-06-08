from model_2 import SweVoice
from dataset import get_feature, CHARS, CommonVoiceDataset
import pydub
import torchaudio
import torch


SAMPLE_PATH = "test_data/test.wav"
CHECKPOINT = "checkpoints/version_2/epoch_37.pt"
waveform, sample_rate = torchaudio.load(SAMPLE_PATH)

n_mels = 80
num_layers = 3
hidden_size = 256
bidirectional = True
reduce_factor = 2 # stride 2

dataset = CommonVoiceDataset('data', 'validated.csv', n_mels=n_mels, reduce_factor=reduce_factor)

model = SweVoice(hidden_size=hidden_size, n_chars=len(CHARS), num_layers=num_layers, input_channels=n_mels, bidirectional=bidirectional)
model.load_state_dict(torch.load(CHECKPOINT))

feature = get_feature(waveform, sample_rate, n_mels)

model.train(False)
with torch.no_grad():
    h, c = model.init_hidden(feature.size(0), 'cpu')
    preds, (_, _) = model(feature, (h, c))
    sample = dataset.translate_from_logits(preds.detach().cpu())[0]

print(sample.replace('*', ''))

