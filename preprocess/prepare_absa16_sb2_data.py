
import json
import sys
import os
import ipdb
from util.clean import clean_text
from glob import glob

root = 'resources/semeval16'


files = [
    'ABSA16_laptops_sb2_test.json',
    'ABSA16_laptops_sb2_train.json',
    'ABSA16_laptops_sb2_trial.json',
    'ABSA16_restaurants_sb2_test.json',
    'ABSA16_restaurants_sb2_train.json',
    'ABSA16_restaurants_sb2_trial.json'
]

for fname in files:
    print(f"\n processing {fname}")

    filename = os.path.join(root, fname)
    dataname, domain, subtask, split = fname.split('.')[0].split('_')
    data = json.load(open(os.path.join(root, fname), 'rt'))
    aspects_term_data = []
    aspects_category_data = []
    for review in data['Reviews']['Review']:

        text_data = []
        aspects_term = []
        aspects_category = []

        if isinstance(review['sentences']['sentence'], list):

            for sentence in review['sentences']['sentence']:
                # ipdb.set_trace()
                text = clean_text(sentence['text'])
                text_data.append(text)
        if isinstance(review['Opinions']['Opinion'], list):
            for opinion in review['Opinions']['Opinion']:
                polarity = opinion['@polarity']
                if '@target' in opinion:
                    term = opinion['@target']
                    aspects_term.append((clean_text(term), clean_text(polarity)))
                if '@category' in opinion:
                    category = opinion['@category']
                    aspects_category.append((clean_text(category), clean_text(polarity)))
        if len(aspects_term) > 0:
            aspects_term_data.append((text_data, aspects_term))
        if len(aspects_category) > 0:
            aspects_category_data.append((text_data, aspects_category))

    save_name = os.path.join(root, f"{dataname}_{domain}_{subtask}_aspect_term_{split}.json")
    print(save_name)
    with open(save_name, 'wt') as f:
        json.dump(aspects_term_data, f, indent=4, sort_keys=True)

    save_name = os.path.join(root, f"{dataname}_{domain}_{subtask}_aspect_category_{split}.json")
    print(save_name)
    with open(save_name, 'wt') as f:
        json.dump(aspects_category_data, f, indent=4, sort_keys=True)
