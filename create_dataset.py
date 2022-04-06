import json
from pathlib import Path
import numpy as np
import pandas as pd
import torchaudio
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser() 
parser.add_argument('--input', type=str, default="./datasets/cv-corpus-8.0-2022-01-19/zh-TW")
args = parser.parse_args()

p = Path(args.input)

def create_json(target="train", path=p, sample_rate_=16000):
    df = pd.read_csv(path / (target + ".tsv"), sep="\t")
    df["path"] = df["path"].str.replace("mp3$", "wav", regex=True)
    # audio_path = tuple(
    #     path / "clips_wav" / df["path"].str.replace("mp3$", "wav", regex=True)
    # )
    # df = df.drop(columns=["path"])

    # df["audio"] = None
    # # l = [None] * len(audio_path)
    # for i, audio in enumerate(tqdm(audio_path[:10])):
    #     waveform, sample_rate = torchaudio.load(audio)
    #     if sample_rate != sample_rate_:
    #         print("altered")
    #         wavefrom = torchaudio.functional.resample(
    #             waveform, sample_rate, sample_rate_
    #         )
    #     df["audio"][i] = {
    #         "path": str(audio),
    #         "array": waveform.numpy()[0],
    #         "sample_rate": sample_rate_,
    #     }
        # l[i] = {
        #     "path": str(audio),
        #     "array": waveform.numpy()[0],
        #     "sample_rate": sample_rate_,
        # }
    # df["audio"] = l
    # del l
    # df = df.drop(['path'], axis=1)

    # print("converting to json")
    df_js = df.to_json(orient="records")
    df_json = json.loads(df_js)
    df_json = {"version": path.parent.name, "data": df_json}

    return df_json

train_json = create_json(target="train")
json_object = json.dumps(train_json, indent=4)
with open(p / "train.json", "w") as f:
    f.write(json_object)

val_json = create_json(target="dev")
json_object = json.dumps(val_json, indent=4)
with open(p / "val.json", "w") as f:
    f.write(json_object)

test_json = create_json(target="test")
json_object = json.dumps(test_json, indent=4)
with open(p / "test.json", "w") as f:
    f.write(json_object)
