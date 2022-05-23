
import os, sys
from glob import glob
import json
import random
import ipdb

resource_dir = '../resources'

files = [
    'semeval14_restaurants_aspect_category_single_train.txt',
    'semeval16_restaurants_sb1_aspect_category_single_train.txt',
    'semeval16_restaurants_sb2_aspect_category_single_train.txt',
    'semeval16_laptops_sb1_aspect_category_single_train.txt',
    'semeval16_laptops_sb2_aspect_category_single_train.txt',
]

shots = [1, 5, 10, 20, 50]

for fname in files:
    print(f"processing {fname}")
    data = open(os.path.join(resource_dir, fname), 'rt').readlines()
    data_dict = {}
    for l in data:
        text = l.strip().split('<|category|>')[0]
        category = l.strip().split('<|category|>')[-1].split('<|endofcategory|>')[0]
        data_dict.setdefault(category, []).append(text)

    for shot in shots:
        data_shot = []
        text_multiple_category_shot = {}
        for category in data_dict:
            if len(data_dict[category]) < shot:
                category_data = data_dict[category]
            else:
                category_data = random.sample(data_dict[category], shot)

            for d in category_data:
                text = f"{d} <|category|> {category} <|endofcategory|> <|endoftext|>"
                data_shot.append(text)

                text_multiple_category_shot.setdefault(d, []).append(category)

        save_name = os.path.join(resource_dir, f"{fname.split('/')[-1].split('.')[0]}_{shot}.txt")
        with open(save_name, 'wt') as f:
            for l in data_shot:
                f.write(f"{l.strip()}\n")

        save_name_new = save_name.replace('single_', '')
        with open(save_name_new, 'wt') as f:
            for l in text_multiple_category_shot:
                text = f"{l} <|category|> {','.join(text_multiple_category_shot[l])} <|endofcategory|> <|endoftext|>"
                f.write(f"{text}\n")

