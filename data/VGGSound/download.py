# pip install yt-dlp
# conda install ffmpeg
# conda install pandas
import argparse
import pandas as pd
import os
import subprocess
import shutil
from tqdm import tqdm

def print_argparse(args: argparse.Namespace) -> None:
    for arg in vars(args):
        print(f'--{arg} = {getattr(args, arg)}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_path', type=str)
    ap.add_argument('--meta_file_path', type=str)
    ap.add_argument('--only_audio', action='store_true')
    args = ap.parse_args()

    print_argparse(args)
    ###########################################

    source_link = 'http://www.robots.ox.ac.uk/~vgg/data/vggsound/vggsound.csv'
    subprocess.call(f'wget {source_link} -O {args.meta_file_path}', shell=True)

    try: 
        if os.path.exists(args.output_path):
            shutil.rmtree(args.output_path)
        os.makedirs(args.output_path)
    except Exception as e:
        raise e

    meta_info = pd.read_csv(args.meta_file_path, sep=',')
    meta_info.columns = ['ID', 'start_sec', 'label', 'split']

    label_dic = {}
    for index, label in enumerate(meta_info['label'].unique()):
        label_dic[label] = index

    label_info = pd.DataFrame(columns=['label', 'description'])
    for k, v in label_dic.items():
        label_info.loc[len(label_info)] = [v, k]
    label_info.to_csv(os.path.join(args.output_path, 'label_dic.csv'), index=False)

    # for row_id, row in tqdm(meta_info.iterrows(), total=meta_info.shape[0]):
    for row_id, row in meta_info.iterrows():
        print(f'#### [{row_id}/{meta_info.shape[0]}]')
        video_id = row['ID']
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        start_time = int(row['start_sec'])
        end_time = start_time + 10
        output_folder = os.path.join(args.output_path, row['split'])
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        if args.only_audio:
            output_file_name = f'{video_id.lstrip('-')}_{start_time:06d}_{label_dic[row['label']]}'
            output_path = os.path.join(output_folder, output_file_name)
            command = f"yt-dlp -x --audio-format wav --download-sections '*{start_time}-{end_time}' '{video_url}' -o '{output_path}'"
        else:
            output_file_name = f'{video_id.lstrip('-')}_{start_time:06d}_{label_dic[row['label']]}.mp4'
            output_path = os.path.join(output_folder, output_file_name)
            command = f"yt-dlp --format bestvideo+bestaudio --merge-output-format mp4 --download-sections '*{start_time}-{end_time}' {video_url} -o '{output_path}'"
        subprocess.call(
            command, 
            shell=True, stdout=subprocess.DEVNULL
        )