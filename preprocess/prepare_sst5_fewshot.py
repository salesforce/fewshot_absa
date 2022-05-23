
import os, sys
from glob import glob
import json
import random
import ipdb

resource_dir = '../resources'
gpt2_dir = 'gpt2'
sst5_dir = 'SST-5'

files = [
    'sst5_train.txt'
]


sst5_labels = {
    'negative': '0',
    'somewhat negative': '1',
    'neutral': '2',
    'somewhat positive': '3',
    'positive':'4'}


shots = [1, 5, 10, 20, 50]

for fname in files:
    print(f"processing {fname}")
    data = open(os.path.join(resource_dir, gpt2_dir, fname), 'rt').readlines()
    data_dict = {}
    for l in data:
        text = l.strip().split('<|sentiment|>')[0].strip()
        sentiment = l.strip().split('<|sentiment|>')[-1].split('<|endofsentiment|>')[0].strip()
        data_dict.setdefault(sentiment, []).append(text)

    for shot in shots:
        data_shot = []
        data_shot_original = []
        csv_data_shot_original = []
        for sentiment in data_dict:
            if len(data_dict[sentiment]) < shot:
                sentiment_data = data_dict[sentiment]
            else:
                sentiment_data = random.sample(data_dict[sentiment], shot)

            for d in sentiment_data:
                text = f"{d} <|sentiment|> {sentiment} <|endofsentiment|> <|endoftext|>"
                data_shot.append(text)

                sentiment_original = sst5_labels[sentiment]
                d_original = d.split('<|review|>')[-1].split('<|endofreview|>')[0].strip()
                text_original = f"{sentiment_original}\t{d_original}"
                data_shot_original.append(text_original)

                d_original_csv = d_original.strip().replace(',', '')
                csv_data_shot_original.append(f"{d_original_csv},{sentiment_original}")

        save_name = os.path.join(resource_dir, gpt2_dir, f"{fname.split('/')[-1].split('.')[0]}_{shot}.txt")
        print(save_name)
        with open(save_name, 'wt') as f:
            for l in data_shot:
                f.write(f"{l.strip()}\n")

        data_dir = sst5_dir
        save_name_original = os.path.join(resource_dir, data_dir, f"train_{shot}.txt")
        print(save_name_original)
        with open(save_name_original, 'wt') as f:
            for l in data_shot_original:
                f.write(f"{l.strip()}\n")

        save_name_original = os.path.join(resource_dir, data_dir, f"train_{shot}.csv")
        print(save_name_original)
        with open(save_name_original, 'wt') as f:
            f.write(f"sentence,label\n")
            for l in csv_data_shot_original:
                f.write(f"{l.strip()}\n")

