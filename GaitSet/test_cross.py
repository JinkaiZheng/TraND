from datetime import datetime
import numpy as np
import argparse

from model.initialization_cross import initialization_cross
from model.utils import evaluation
from config import conf_CASIA, conf_OULP


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--iter', default='100000', type=int,
                    help='iter: iteration of the checkpoint to load. Default: 80000')
parser.add_argument('--batch_size', default='64', type=int,
                    help='batch_size: batch size for parallel test. Default: 64')
parser.add_argument('--source', default='casia-b', type=str,
                    help='source dataset to be used. (default: casia-b)')
parser.add_argument('--target', default='oulp', type=str,
                    help='target dataset to be used. (default: oulp)')
parser.add_argument('--cache', default=False, type=boolean_string,
                    help='cache: if set as TRUE all the test data will be loaded at once'
                         ' before the transforming start. Default: FALSE')
opt = parser.parse_args()


# Exclude identical-view cases
def de_diag(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1)
    result = result / (result.shape[0]-1.0)
    if not each_angle:
        result = np.mean(result)
    return result


if opt.source == "casia-b":
    conf_source = conf_CASIA
elif opt.source == "oulp":
    conf_source = conf_OULP
else:
    raise Warning("Please check your source/target dataset, current dataset is not casia-b or oulp.")

if opt.target == "casia-b":
    conf_target = conf_CASIA
elif opt.target == "oulp":
    conf_target = conf_OULP
else:
    raise Warning("Please check your target dataset, current dataset is not casia-b or oulp.")

m = initialization_cross(conf_source, conf_target, test=opt.cache)[0]

# load model checkpoint of iteration opt.iter
print('Loading the model of iteration %d...' % opt.iter)
m.load(opt.iter)
print('Transforming...')
time = datetime.now()
test = m.transform('test', opt.batch_size)
print('Evaluating...')
acc = evaluation(test, conf_target['data'])
print('Evaluation complete. Cost:', datetime.now() - time)

if acc.shape[0] == 3:
    for i in range(1):
        print('===Rank-%d (Include identical-view cases) on the CASIA-B dataset===' % (i + 1))
        print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
            np.mean(acc[0, :, :, i]),
            np.mean(acc[1, :, :, i]),
            np.mean(acc[2, :, :, i])))

    for i in range(1):
        print('===Rank-%d (Exclude identical-view cases) on the CASIA-B dataset===' % (i + 1))
        print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
            de_diag(acc[0, :, :, i]),
            de_diag(acc[1, :, :, i]),
            de_diag(acc[2, :, :, i])))

    np.set_printoptions(precision=2, floatmode='fixed')
    for i in range(1):
        print('===Rank-%d of each angle (Exclude identical-view cases) on the CASIA-B dataset===' % (i + 1))
        print('NM:', de_diag(acc[0, :, :, i], True))
        print('BG:', de_diag(acc[1, :, :, i], True))
        print('CL:', de_diag(acc[2, :, :, i], True))
elif acc.shape[0] == 1:
    for i in range(1):
        print('===Rank-%d (Include identical-view cases) on the OULP dataset===' % (i + 1))
        print('NM: %.3f' % (
            np.mean(acc[0, :, :, i])))

    for i in range(1):
        print('===Rank-%d (Exclude identical-view cases) on the OULP dataset===' % (i + 1))
        print('NM: %.3f' % (
            de_diag(acc[0, :, :, i])))

    np.set_printoptions(precision=2, floatmode='fixed')
    for i in range(1):
        print('===Rank-%d of each angle (Exclude identical-view cases) on the OULP dataset===' % (i + 1))
        print('NM:', de_diag(acc[0, :, :, i], True))
