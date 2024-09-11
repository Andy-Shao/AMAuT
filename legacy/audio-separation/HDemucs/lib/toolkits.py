import torch
import torch.nn as nn

def separate_sources(model:nn.Module, mix:torch.Tensor, sample_rate:int, segment=10.0, overlap=0.1, device=None):
    """
    Apply model to a given mixture. Use fade, and add segments together in order to add model segment by segment.

    Args:
        segment (int): segment length in seconds
        device (torch.device, str, or None): if provided, device on which to
            execute the computation, otherwise `mix.device` is assumed.
            When `device` is different from `mix.device`, only local computations will
            be on `device`, while the entire tracks will be stored on `mix.device`.
    """
    from torchaudio.transforms import Fade

    if device is None:
        device = mix.device
    else: 
        device = torch.device(device)
    
    batch, channels, length = mix.shape

    chunk_len = int(sample_rate * segment * (1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = overlap * sample_rate
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")

    final = torch.zeros(batch, len(model.sources), channels, length, device=device)

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model.forward(chunk)
        out = fade(out)
        final[:, :, :, start:end] += out
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0
    return final

def plot_spectrogram(stft, title="Spectrogram"):
    import matplotlib.pyplot as plt
    magnitude = stft.abs()
    spectrogram = 20 * torch.log10(magnitude + 1e-8).numpy()
    _, axis = plt.subplots(1, 1)
    axis.imshow(spectrogram, cmap="viridis", vmin=-60, vmax=0, origin="lower", aspect="auto")
    axis.set_title(title)
    plt.tight_layout()

def output_results(original_source: torch.Tensor, predicated_source: torch.Tensor, source:str, tsf:nn.Module):
    from mir_eval import separation
    print(
        'SDR score is: ',
        separation.bss_eval_sources(original_source.detach().numpy(), predicated_source.detach().numpy())[0].mean())
    plot_spectrogram(tsf(predicated_source)[0], f'Spectrogram - {source}')