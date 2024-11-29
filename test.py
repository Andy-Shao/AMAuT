import soundfile as sf

from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch 
import torchaudio

from lib.scDataset import SpeechCommandsDataset

dataset = SpeechCommandsDataset(root_path='/home/andyshao/data/speech_commands', include_rate=True, data_type='all')

wavform, label, sample_rate = dataset[0]
if sample_rate != 16000:
    wavform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(wavform)
wavform = wavform / torch.abs(wavform).max()

# facebook/wav2vec2-base-960h
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')

# Preprocess and forward pass
inputs = processor(wavform.squeeze().numpy(), sample_rate=16000, return_tensors='pt', padding=True)
with torch.no_grad():
    outputs = model(**inputs)

# Extract embeddings
embeddings = outputs.last_hidden_state # Shape: [batch_size, sequence_length, hidden_size]
pooled_embeddings = torch.mean(embeddings, dim=1) # Shape: [batchsize, hidden_size]
print(f'embeddings shape:{embeddings.shape}, pooled_embeddings shape:{pooled_embeddings.shape}')