
import os, sys
from glob import glob
import json
import random

resource_dir = 'resources'

files = [
    'semeval14_laptops_aspect_term_single_ss_train.txt',
    'semeval14_restaurants_aspect_term_single_ss_train.txt',
    'semeval16_restaurants_sb1_aspect_term_single_ss_train.txt',
]

shots = [0.01, 0.05, 0.1, 0.2, 0.5]

for model_dir in ['gpt2']:
    for fname in files:
        filename = os.path.join(resource_dir, model_dir, fname)
        print(f"processing {filename}")
        data = open(filename, 'rt').readlines()
        for shot in shots:
            num_data = int(shot * len(data))
            random.shuffle(data)
            data_shot = data[:num_data]
            save_name = os.path.join(resource_dir, model_dir, f"{fname.split('/')[-1].split('.')[0]}_{shot}.txt")
            with open(save_name, 'wt') as f:
                for l in data_shot:
                    f.write(f"{l.strip()}\n")
