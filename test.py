from pytube import YouTube

video_id = '---g-f_I2yQ'
yt = YouTube("https://www.youtube.com/watch?v=" + video_id)
stream = yt.streams.get_highest_resolution()
file_path = stream.download(output_path='./')
print(file_path)

# pip install yt-dlp
# conda install ffmpeg

# yt-dlp --format bestvideo+bestaudio --merge-output-format mp4 -o "videos/%(id)s.%(ext)s" https://www.youtube.com/watch?v=---g-f_I2yQ

# yt-dlp --format bestvideo+bestaudio --merge-output-format mp4 -o "videos/%(id)s.%(ext)s" https://www.youtube.com/watch?v=---g-f_I2yQ