
import json
import os, sys
import ipdb

data_dir = 'resources/semeval16'

for split in ['train', 'test', 'trial']:
    for domain in ['restaurants', 'laptops']:
        sb1_term = json.load(open(os.path.join(data_dir, f"ABSA16_{domain}_sb1_aspect_term_{split}.json"), 'rt'))
        sb2_category = json.load(open(os.path.join(data_dir, f"ABSA16_{domain}_sb2_aspect_category_{split}.json"), 'rt'))
        sb1_term_dic = {}
        for s, t in sb1_term:
            sb1_term_dic[s] = t

        sb2_term = []
        for text, _ in sb2_category:
            tmp = []
            for t in text:
                if t in sb1_term_dic:
                    # tmp.append(sb1_term_dic[t][0])
                    # ipdb.set_trace()
                    # tmp.extend(sb1_term_dic[t])
                    for term in sb1_term_dic[t]:
                        if term not in tmp:
                            tmp.append(term)

            sb2_term.append((text, tmp))

        save_name = os.path.join(data_dir, f"ABSA16_{domain}_sb2_aspect_term_{split}.json")
        print(save_name)
        with open(save_name, 'wt') as f:
            json.dump(sb2_term, f, indent=4, sort_keys=True)