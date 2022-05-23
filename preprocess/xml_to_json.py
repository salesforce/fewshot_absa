
import xmltodict
import json
from glob import glob
import sys, os


dir_name = sys.argv[1]
filenames = glob(os.path.join(dir_name, '*xml'))

for fname in filenames:
    o = xmltodict.parse(open(fname, 'rt').read())
    fn = fname.split('/')[-1].split('.')[0]
    save_name = os.path.join(dir_name, f"{fn}.json")
    print(save_name)
    with open(save_name, 'wt') as f:
        json.dump(o, f, indent=4, sort_keys=True)
