from pathlib import Path
import numpy as np
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser() 
parser.add_argument('--input', type=str, default="./datasets/cv-corpus-8.0-2022-01-19/zh-TW")
# parser.add_argument('--output', type=str, default="./datasets/cv-corpus-8.0-2022-01-19/zh-TW")
args = parser.parse_args()


p = Path(args.input)
p_new = p / "clips_wav"
if not p_new.exists():
    p_new.mkdir(parents=True, exist_ok=False)
    
train = pd.read_csv(p / "train.tsv", sep="\t")
dev = pd.read_csv(p / "dev.tsv", sep="\t")
test = pd.read_csv(p / "test.tsv", sep="\t")

mp3_files = [
    p / "clips" / x for df in [train["path"], dev["path"], test["path"]] for x in df
]

for file in tqdm(mp3_files):
    sound = AudioSegment.from_mp3(file)
    sound = sound.set_frame_rate(16000)
    out = sound.export(p_new / (file.stem + ".wav"), format="wav")
    out.close()