
import json
import os, sys
from glob import glob
import ipdb

data_dir = '../resources'
gpt2_dir = 'gpt2'

# preprocess semeval14
data_name = 'semeval14'
for domain in ['restaurants', 'laptops']:
    for split in ['train', 'test', 'trial']:
        for task in ['aspect_category', 'aspect_term']:
            filename = os.path.join(data_dir, data_name, f"{domain}_{task}_{split}.json")
            print(f"processing {filename}")
            data = json.load(open(filename, 'rt'))
            sequence_data = []
            for text, target in data:
                if len(target) == 0:
                    continue
                task_token = '<|term|>' if task == 'aspect_term' else '<|category|>'
                task_end_token = '<|endofterm|>' if task == 'aspect_term' else '<|endofcategory|>'
                target_seq = []
                for trg in target:
                    tmp = ' '.join(trg)
                    if tmp not in target_seq:
                        target_seq.append(tmp)

                for trg_seq in target_seq:
                    text_sequence = f"<|endoftext|> <|review|> {text} <|endofreview|> {task_token} {trg_seq} {task_end_token} <|endoftext|>"
                    sequence_data.append(text_sequence)

            save_name = filename = os.path.join(data_dir, gpt2_dir, f"{data_name}_{domain}_{task}_single_{split}.txt")
            with open(save_name, 'wt') as f:
                for seq in sequence_data:
                    f.write(f"{seq}\n")


# preprocess semeval16
data_name = 'semeval16'
for domain in ['restaurants', 'laptops']:
    for split in ['train', 'test', 'trial']:
        for target_name in ['aspect_category', 'aspect_term']:
            for task in ['sb1', 'sb2']:
                filename = os.path.join(data_dir, data_name, f"ABSA16_{domain}_{task}_{target_name}_{split}.json")
                print(f"processing {filename}")
                data = json.load(open(filename, 'rt'))
                sequence_data = []

                if task == 'sb2':
                    sequence_data_new = []

                for text, target in data:
                    if len(target) == 0:
                        continue
                    task_token = '<|term|>' if target_name == 'aspect_term' else '<|category|>'
                    task_end_token = '<|endofterm|>' if target_name == 'aspect_term' else '<|endofcategory|>'
                    target_seq = []

                    for trg in target:
                        tmp = ' '.join(trg)
                        if tmp not in target_seq:
                            target_seq.append(tmp)

                    if task == 'sb2':
                        text_joined = ' '.join(text)
                    else:
                        text_joined = text
                    for trg_seq in target_seq:
                        text_sequence = f"<|endoftext|> <|review|> {text_joined} <|endofreview|> {task_token} {trg_seq} {task_end_token} <|endoftext|>"
                        sequence_data.append(text_sequence)

                    if task == 'sb2':
                        text_joined = '<|sentence|> ' + ' <|sentence|> '.join(text)
                        for trg_seq in target_seq:
                            text_sequence = f"<|endoftext|> <|review|> {text_joined} <|endofreview|> {task_token} {trg_seq} {task_end_token} <|endoftext|>"
                            sequence_data_new.append(text_sequence)
                save_name = filename = os.path.join(data_dir, gpt2_dir, f"{data_name}_{domain}_{task}_{target_name}_single_{split}.txt")
                with open(save_name, 'wt') as f:
                    for seq in sequence_data:
                        f.write(f"{seq}\n")

                if task == 'sb2':
                    save_name = filename = os.path.join(data_dir, gpt2_dir,
                                                        f"{data_name}_{domain}_{task}_{target_name}_single_new_{split}.txt")
                    with open(save_name, 'wt') as f:
                        for seq in sequence_data_new:
                            f.write(f"{seq}\n")