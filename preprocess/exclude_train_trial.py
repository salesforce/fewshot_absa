

import json
import ipdb

resource_dir = 'resources'

for year in ['semeval14']:
    for domain in ['restaurants', 'laptops']:
        for task in ['aspect_term', 'aspect_category']:

            train = json.load(open(f"{resource_dir}/{year}/{domain}_{task}_train.json", 'rt'))
            trial = json.load(open(f"{resource_dir}/{year}/{domain}_{task}_trial.json", 'rt'))
            test = json.load(open(f"{resource_dir}/{year}/{domain}_{task}_test.json", 'rt'))

            for example in test:
                if example in train:
                    train.remove(example)

            train_new = []
            for example in train:
                if example in trial:
                    continue
                train_new.append(example)


            with open(f"{resource_dir}/{year}/{domain}_{task}_train.json", 'wt') as f:
                json.dump(train_new, f, indent=4, sort_keys=True)


            flag = True
            count = 0
            for example in test:
                if example in train or example in trial or example in train_new:
                    flag = False
                    count += 1

            total = len(train) + len(test) + len(trial)
            print(flag)
            print(count)
            print('train', len(train_new), len(train)/total)
            print('trial', len(trial), len(trial)/total)
            print('test', len(trial), len(test)/total)
