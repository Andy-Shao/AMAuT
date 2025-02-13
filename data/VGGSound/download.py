# pip install yt-dlp
# conda install ffmpeg
import argparse

from lib.toolkit import print_argparse

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ouput_path', type=str)
    ap.add_argument('--meta_file_path', type=str)
    args = ap.parse_args()

    print_argparse(args)
    ###########################################