# from pytube import YouTube

# video_id = '---g-f_I2yQ'
# yt = YouTube("https://www.youtube.com/watch?v=" + video_id)
# stream = yt.streams.get_highest_resolution()
# file_path = stream.download(output_path='./')
# print(file_path)

# pip install yt-dlp
# conda install ffmpeg

# yt-dlp --format bestvideo+bestaudio --merge-output-format mp4 -o "videos/%(id)s.%(ext)s" https://www.youtube.com/watch?v=---g-f_I2yQ

# yt-dlp --format mp4 --download-sections "*${start_time}-${end_time}" "https://www.youtube.com/watch?v=${video_id}" -o "vggsound_videos/${video_id}.mp4"

# yt-dlp --format mp4 --download-sections "*1-11" "https://www.youtube.com/watch?v=---g-f_I2yQ" -o 'my.mp4'

# import yt_dlp
import os

video_id = '--5OkAjCI7g'
video_url = f"https://www.youtube.com/watch?v={video_id}"
start_time = 1
end_time = start_time + 10
output_folder = './'
output_file_name = f'{video_id.lstrip('-')}_{start_time:06d}.mp4'
output_path = os.path.join(output_folder, )

# ydl_opts = {
#     'format': 'mp4',
#     'paths': {'home': output_path},
#     'outtmpl': output_file_name,
#     'download_sections': { '*': '00:00:01-00:00:11' },
#     'quiet': False
# }

# with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#     try:
#         ydl.download([video_url])
#         print(f"Downloaded: {output_path}")
#     except Exception as e:
#         print(f"Failed to download {video_id}: {e}")

# import subprocess

# subprocess.call(f"yt-dlp --format bestvideo+bestaudio --merge-output-format mp4 --download-sections '*{start_time}-{end_time}' {video_url} -o '{output_file_name}'", shell=True)

# print("Download complete!")

import pandas as pd

meta_info = pd.read_csv('/root/vggsound.csv', sep=',')
meta_info.columns = ['ID', 'start_sec', 'label', 'split']
print(meta_info['label'].unique())
