from model.initialization import initialization
from config import conf_CASIA, conf_OULP
import argparse


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--data', default='casia-b', type=str,
                    help='dataset to be used. (default: casia-b)')
parser.add_argument('--cache', default=False, type=boolean_string,
                    help='cache: if set as TRUE all the training data will be loaded at once'
                         ' before the training start. Default: False')
opt = parser.parse_args()

if opt.data == "casia-b":
    conf = conf_CASIA
elif opt.data == "oulp":
    conf = conf_OULP
else:
    raise Warning("Please check your dataset_name, current dataset is not casia-b or oulp.")

m = initialization(conf, train=opt.cache)[0]

print("Training START")
m.fit()
print("Training COMPLETE")
