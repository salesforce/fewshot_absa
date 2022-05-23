
import json
import os, sys
from glob import glob
import ipdb

data_dir = '../resources'
gpt2_dir = 'gpt2'

# preprocess semeval14
for data_name in ['semeval14']:
    for domain in ['restaurants']:
        for split in ['train', 'test']:
            filename_term = os.path.join(data_dir, data_name, f"{domain}_aspect_term_{split}.json")
            filename_category = os.path.join(data_dir, data_name, f"{domain}_aspect_category_{split}.json")
            data_term = json.load(open(filename_term, 'rt'))
            data_category = json.load(open(filename_category, 'rt'))
            data_dict = {}
            for text, target in data_term:
                data_dict[text] = {}
                data_dict[text]['term'] = target

            for text, target in data_category:
                if text not in data_dict:
                    data_dict[text] = {}
                    data_dict[text]['category'] = target
                else:
                    data_dict[text]['category'] = target

            sequence_data = []
            target_term, target_category = None, None
            for text in data_dict:
                if 'term' in data_dict[text]:
                    target_term = data_dict[text]['term']
                if 'category' in data_dict[text]:
                    target_category = data_dict[text]['category']

                term_start_token = '<|term|>'
                term_end_token = '<|endofterm|>'
                category_start_token = '<|category|>'
                category_end_token = '<|endofcategory|>'

                if target_term is None:
                    term_target_seq = ''
                else:
                    term_target_seq = []
                    for trg in target_term:
                        tmp = ' '.join(trg)
                        if tmp not in term_target_seq:
                            term_target_seq.append(tmp)
                    term_target_seq = ' , '.join(term_target_seq)

                if target_category is None:
                    category_target_seq = ''
                else:
                    category_target_seq = []
                    for trg in target_category:
                        tmp = ' '.join(trg)
                        if tmp not in category_target_seq:
                            category_target_seq.append(tmp)
                    category_target_seq = ' , '.join(category_target_seq)

                text_sequence = f"<|endoftext|> <|review|> {text} <|endofreview|> <|term|> {term_target_seq} <|endofterm|> <|category|> {category_target_seq} <|endofcategory|> <|endoftext|>"
                sequence_data.append(text_sequence)
            save_name = filename = os.path.join(data_dir, gpt2_dir, f"{data_name}_{domain}_aspect_term_aspect_category_{split}.txt")
            with open(save_name, 'wt') as f:
                for seq in sequence_data:
                    f.write(f"{seq}\n")


for data_name in ['semeval16']:
    for domain in ['restaurants']:
        for task in ['sb1', 'sb2']:
            for split in ['train', 'test']:

                filename_term = os.path.join(data_dir, data_name, f"ABSA16_{domain}_{task}_aspect_term_{split}.json")
                filename_category = os.path.join(data_dir, data_name, f"ABSA16_{domain}_{task}_aspect_category_{split}.json")
                data_term = json.load(open(filename_term, 'rt'))
                data_category = json.load(open(filename_category, 'rt'))
                data_dict = {}
                for text, target in data_term:
                    if task == 'sb2':
                        text = ' '.join(text)
                    data_dict[text] = {}
                    data_dict[text]['term'] = target

                for text, target in data_category:
                    if task == 'sb2':
                        text = ' '.join(text)
                    if text not in data_dict:
                        data_dict[text] = {}
                        data_dict[text]['category'] = target
                    else:
                        data_dict[text]['category'] = target

                sequence_data = []
                target_term, target_category = None, None
                for text in data_dict:
                    if 'term' in data_dict[text]:
                        target_term = data_dict[text]['term']
                    if 'category' in data_dict[text]:
                        target_category = data_dict[text]['category']

                    term_start_token = '<|term|>'
                    term_end_token = '<|endofterm|>'
                    category_start_token = '<|category|>'
                    category_end_token = '<|endofcategory|>'

                    if target_term is None:
                        term_target_seq = ''
                    else:
                        term_target_seq = []
                        for trg in target_term:
                            tmp = ' '.join(trg)
                            if tmp not in term_target_seq:
                                term_target_seq.append(tmp)
                        term_target_seq = ' , '.join(term_target_seq)

                    if target_category is None:
                        category_target_seq = ''
                    else:
                        category_target_seq = []
                        for trg in target_category:
                            tmp = ' '.join(trg)
                            if tmp not in category_target_seq:
                                category_target_seq.append(tmp)
                        category_target_seq = ' , '.join(category_target_seq)

                    text_sequence = f"<|endoftext|> <|review|> {text} <|endofreview|> <|term|> {term_target_seq} <|endofterm|> <|category|> {category_target_seq} <|endofcategory|> <|endoftext|>"
                    sequence_data.append(text_sequence)

                save_name = filename = os.path.join(data_dir, gpt2_dir, f"{data_name}_{domain}_{task}_aspect_term_aspect_category_{split}.txt")
                with open(save_name, 'wt') as f:
                    for seq in sequence_data:
                        f.write(f"{seq}\n")


