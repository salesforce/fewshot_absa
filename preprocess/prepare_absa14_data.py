
import json
import sys
import os
import ipdb
from util.clean import clean_text
from glob import glob

root = 'resources'
data_dir = 'semeval14'
# gpt2_dir = 'gpt2'
# t5_dir = 't5'

files = [
    'laptops_train.json',
    'laptops_trial.json',
    'laptops_test.json',
    'restaurants_train.json',
    'restaurants_trial.json',
    'restaurants_test.json',
]

# root = sys.argv[1]
# files = glob(os.path.join(root, '*json'))


for fname in files:
    filename = os.path.join(root, data_dir, fname)
    dataname = fname.split('.')[0].split('_')[0]
    split = '_'.join(fname.split('.')[0].split('_')[1:])
    data = json.load(open(filename, 'rt'))
    text_aspect_term_data = []
    text_aspect_category_data = []

    for example in data['sentences']['sentence']:
        text = clean_text(example['text'])
        aspects_term_data = []
        aspects_category_data = []

        # aspect category
        if 'aspectTerms' in example:
            if isinstance(example['aspectTerms']['aspectTerm'], list):
                for aspect in example['aspectTerms']['aspectTerm']:
                    try:
                        term = aspect['@term']
                        polarity = aspect['@polarity']
                        aspects_term_data.append((clean_text(term), clean_text(polarity)))
                    except:
                        ipdb.set_trace()
            else:
                try:
                    aspect = example['aspectTerms']['aspectTerm']
                    term = aspect['@term']
                    polarity = aspect['@polarity']
                    aspects_term_data.append((clean_text(term), clean_text(polarity)))
                except:
                    ipdb.set_trace()

        # aspect category
        if 'aspectCategories' in example:
            if isinstance(example['aspectCategories']['aspectCategory'], list):
                for aspect in example['aspectCategories']['aspectCategory']:
                    try:
                        term = aspect['@category']
                        polarity = aspect['@polarity']
                        aspects_category_data.append((clean_text(term), clean_text(polarity)))
                    except:
                        ipdb.set_trace()
            else:
                try:
                    aspect = example['aspectCategories']['aspectCategory']
                    term = aspect['@category']
                    polarity = aspect['@polarity']
                    aspects_category_data.append((clean_text(term), clean_text(polarity)))
                except:
                    ipdb.set_trace()

        text_aspect_term_data.append((text, aspects_term_data))
        text_aspect_category_data.append((text, aspects_category_data))

    with open(os.path.join(root, data_dir, f"{dataname}_aspect_term_{split}.json"), 'wt') as f:
        json.dump(text_aspect_term_data, f, indent=4, sort_keys=True)

    with open(os.path.join(root, data_dir, f"{dataname}_aspect_category_{split}.json"), 'wt') as f:
        json.dump(text_aspect_category_data, f, indent=4, sort_keys=True)

    with open(os.path.join(root, data_dir, f"{dataname}_aspect_term_{split}.json"), 'wt') as f:
        json.dump(text_aspect_term_data, f, indent=4, sort_keys=True)

    with open(os.path.join(root, data_dir, f"{dataname}_aspect_category_{split}.json"), 'wt') as f:
        json.dump(text_aspect_category_data, f, indent=4, sort_keys=True)
