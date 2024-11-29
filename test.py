import soundfile as sf

from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch 

processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')

audio, sample_rate = sf.read(...)
input_values = processor(audio, return_tensors='pt', sampling_rate=16000).input_values
with torch.no_grad():
    embeddings = model(input_values).last_hidden_state