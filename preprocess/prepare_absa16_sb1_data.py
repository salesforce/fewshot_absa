
import json
import sys
import os
import ipdb
from util.clean import clean_text
from glob import glob

root = 'resources/semeval16'


files = [
    'ABSA16_laptops_sb1_test.json',
    'ABSA16_laptops_sb1_train.json',
    'ABSA16_laptops_sb1_trial.json',
    'ABSA16_laptops_sb2_test.json',
    'ABSA16_laptops_sb2_train.json',
    'ABSA16_laptops_sb2_trial.json',
    'ABSA16_restaurants_sb1_test.json',
    'ABSA16_restaurants_sb1_train.json',
    'ABSA16_restaurants_sb1_trial.json',
    'ABSA16_restaurants_sb2_test.json',
    'ABSA16_restaurants_sb2_train.json',
    'ABSA16_restaurants_sb2_trial.json'
]

for fname in files:
    if 'sb2' in fname:
        print(f"sb2, skipping {fname}")
        continue
    print(f"\n processing {fname}")

    filename = os.path.join(root, fname)
    dataname, domain, subtask, split = fname.split('.')[0].split('_')
    data = json.load(open(os.path.join(root, fname), 'rt'))
    aspects_term_data = []
    aspects_category_data = []
    aspects_out_of_scope = []

    for review in data['Reviews']['Review']:
        if isinstance(review['sentences']['sentence'], list):

            for example in review['sentences']['sentence']:

                text = clean_text(example['text'])

                # test set out of scope
                if '@OutOfScope' in example and example['@OutOfScope'] == "TRUE" and 'test' in split:
                    aspects_out_of_scope.append(text)
                    continue

                aspects_term = []
                aspects_category = []
                if 'Opinions' in example and example['Opinions'] is not None:
                    if isinstance(example['Opinions']['Opinion'], list):
                        for opinion in example['Opinions']['Opinion']:
                            polarity = opinion['@polarity']
                            if '@target' in opinion:
                                if opinion['@target'] != 'NULL':
                                    term = opinion['@target']
                                    aspects_term.append((clean_text(term), clean_text(polarity)))
                            if '@category' in opinion:
                                category = opinion['@category']
                                aspects_category.append((clean_text(category), clean_text(polarity)))
                    else:

                        opinion = example['Opinions']['Opinion']
                        polarity = opinion['@polarity']
                        if '@target' in opinion:
                            if opinion['@target'] != 'NULL':
                                term = opinion['@target']
                                aspects_term.append((clean_text(term), clean_text(polarity)))
                        if '@category' in opinion:
                            category = opinion['@category']
                            aspects_category.append((clean_text(category), clean_text(polarity)))
                if len(aspects_term) > 0:
                    aspects_term = list(set(aspects_term))
                    aspects_term_data.append((text, aspects_term))
                if len(aspects_category) > 0:
                    aspects_category_data.append((text, aspects_category))
        else:
            example = review['sentences']['sentence']
            text = clean_text(example['text'])
            aspects_term = []
            aspects_category = []
            if 'Opinions' in example and example['Opinions'] is not None:
                if isinstance(example['Opinions']['Opinion'], list):
                    for opinion in example['Opinions']['Opinion']:
                        polarity = opinion['@polarity']
                        if '@target' in opinion:
                            if opinion['@target'] != 'NULL':
                                term = opinion['@target']
                                aspects_term.append((clean_text(term), clean_text(polarity)))
                        if '@category' in opinion:
                            category = opinion['@category']
                            aspects_category.append((clean_text(category), clean_text(polarity)))
                else:
                    opinion = example['Opinions']['Opinion']
                    polarity = opinion['@polarity']
                    if '@target' in opinion:
                        if opinion['@target'] != 'NULL':
                            term = opinion['@target']
                            aspects_term.append((clean_text(term), clean_text(polarity)))
                    if '@category' in opinion:
                        category = opinion['@category']
                        aspects_category.append((clean_text(category), clean_text(polarity)))
            if len(aspects_term) > 0:
                aspects_term = list(set(aspects_term))
                aspects_term_data.append((text, aspects_term))
            if len(aspects_category) > 0:
                aspects_category_data.append((text, aspects_category))

    with open(os.path.join(root, f"{dataname}_{domain}_{subtask}_aspect_term_{split}.json"), 'wt') as f:
        json.dump(aspects_term_data, f, indent=4, sort_keys=True)
    with open(os.path.join(root, f"{dataname}_{domain}_{subtask}_aspect_category_{split}.json"), 'wt') as f:
        json.dump(aspects_category_data, f, indent=4, sort_keys=True)

    if 'test' in split and len(aspects_out_of_scope) > 0:
        with open(os.path.join(root, f"{dataname}_{domain}_{subtask}_{split}_oos.json"), 'wt') as f:
            json.dump(aspects_out_of_scope, f, indent=4, sort_keys=True)
