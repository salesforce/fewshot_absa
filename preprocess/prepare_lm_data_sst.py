
import json
import os, sys
from glob import glob
import ipdb
import csv

data_dir = 'resources'
gpt2_dir = 'gpt2'

sst2_labels = {
    '0': 'negative',
    '1': 'positive'}

sst5_labels = {
    '0': 'negative',
    '1': 'somewhat negative',
    '2': 'neutral',
    '3': 'somewhat positive',
    '4': 'positive'}


## Prepare SST-2 ##
data_name = 'SST-2'
target_data_name = 'sst2'
sst2_dir = 'SST-2'
for split in ['train', 'test', 'dev']:
        filename = os.path.join(data_dir, data_name, f"{split}.tsv")
        print(f"processing {filename}")
        data = csv.reader(open(filename, 'r'), delimiter='\t')
        sequence_data = []
        sequence_data_t5 = []
        csv_data = []
        json_data = []
        if split == 'test':
            for _, text in data:
                if text.strip() == 'sentence':
                    continue
                task_token = '<|sentiment|>'
                task_end_token = '<|endofsentiment|>'
                text_sequence = f"<|endoftext|> <|review|> {text.strip()} <|endofreview|> {task_token}"
                sequence_data.append(text_sequence)
                csv_data.append(f"{text.strip()}")
        else:

            for text, target in data:
                if target.strip() == 'label':
                    continue
                task_token = '<|sentiment|>'
                task_end_token = '<|endofsentiment|>'
                target_seq = sst2_labels[target]
                text_sequence = f"<|endoftext|> <|review|> {text.strip()} <|endofreview|> {task_token} {target_seq} {task_end_token} <|endoftext|>"
                sequence_data.append(text_sequence)
                csv_text = text.strip().replace(',', '')
                csv_data.append(f"{csv_text},{target.strip()}")
                json_data.append({"sentence": text.strip(),
                                  "label": target.strip()})

        save_name = os.path.join(data_dir, gpt2_dir, f"{target_data_name}_{split}.txt")
        with open(save_name, 'wt') as f:
            for seq in sequence_data:
                f.write(f"{seq}\n")

        save_name = os.path.join(data_dir, sst2_dir, f"{split}.csv")
        with open(save_name, 'wt') as f:
            f.write(f"sentence,label\n")
            for seq in csv_data:
                f.write(f"{seq}\n")

        save_name = os.path.join(data_dir, sst2_dir, f"{split}.txt")
        with open(save_name, 'wt') as f:
            for seq in json_data:
                f.write(f"{seq}\n")


## Prepare SST-5 ##
data_name = 'SST-5'
target_data_name = 'sst5'
sst5_dir = 'SST-5'
for split in ['train', 'test', 'dev']:
        filename = os.path.join(data_dir, data_name, f"{split}.txt")
        print(f"processing {filename}")
        data = csv.reader(open(filename, 'r'), delimiter='\t')
        sequence_data = []
        sequence_data_t5 = []
        csv_data = []
        json_data = []
        for target, text in data:
            if target.strip() == 'label':
                continue
            task_token = '<|sentiment|>'
            task_end_token = '<|endofsentiment|>'
            target_seq = sst5_labels[target]
            text_sequence = f"<|endoftext|> <|review|> {text.strip().lower()} <|endofreview|> {task_token} {target_seq.lower()} {task_end_token} <|endoftext|>"
            sequence_data.append(text_sequence)
            csv_text = text.strip().replace(',', '')
            csv_data.append(f"{csv_text},{target.strip()}")
            json_data.append({'sentence': text.strip(),
                              'label': target.strip()})

        save_name = os.path.join(data_dir, gpt2_dir, f"{target_data_name}_{split}.txt")
        with open(save_name, 'wt') as f:
            for seq in sequence_data:
                f.write(f"{seq}\n")

        save_name = os.path.join(data_dir, sst5_dir, f"{split}.csv")
        with open(save_name, 'wt') as f:
            f.write(f"sentence,label\n")
            for seq in csv_data:
                f.write(f"{seq}\n")

        save_name = os.path.join(data_dir, sst5_dir, f"{split}.csv")
        with open(save_name, 'wt') as f:
            f.write(f"sentence,label\n")
            for seq in csv_data:
                f.write(f"{seq}\n")



