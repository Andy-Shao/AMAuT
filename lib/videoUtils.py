from moviepy.editor import VideoFileClip

import torch

def get_audio_from_video(video_path:str, sample_rate:int) -> torch.Tensor:
    # clip = VideoFileClip(video_path)
    with VideoFileClip(video_path) as clip:
        audio = clip.audio.to_soundarray(fps=sample_rate)  # Converts to a NumPy array
    # clip.close()
    waveform = torch.tensor(audio.T)
    return waveform