# -*- coding: utf-8 -*-
# @Author  : Abner
# @Time    : 2018/12/19

import os
import os.path as osp
import imageio
import cv2
import numpy as np
from warnings import warn
from time import sleep
import argparse

from multiprocessing import Pool
from multiprocessing import TimeoutError as MP_TimeoutError

START = "START"
FINISH = "FINISH"
WARNING = "WARNING"
FAIL = "FAIL"


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--input_path', default='', type=str,
                    help='Root path of raw dataset.')
parser.add_argument('--output_path', default='./data/OULP', type=str,
                    help='Root path for output.')
parser.add_argument('--log_file', default='./pretreatment_oulp.log', type=str,
                    help='Log file path. Default: ./pretreatment_oulp.log')
parser.add_argument('--log', default=False, type=boolean_string,
                    help='If set as True, all logs will be saved. '
                         'Otherwise, only warnings and errors will be saved.'
                         'Default: False')
parser.add_argument('--worker_num', default=20, type=int,
                    help='How many subprocesses to use for data pretreatment. '
                         'Default: 20')
opt = parser.parse_args()

INPUT_PATH = osp.join(opt.input_path, "OULP-C1V1_NormalizedSilhouette")
OUTPUT_PATH = opt.output_path
IF_LOG = opt.log
LOG_PATH = opt.log_file
WORKERS = opt.worker_num

T_H = 64
T_W = 64

def log2str(pid, comment, logs):
    str_log = ''
    if type(logs) is str:
        logs = [logs]
    for log in logs:
        str_log += "# JOB %d : --%s-- %s\n" % (
            pid, comment, log)
    return str_log


def log_print(pid, comment, logs):
    str_log = log2str(pid, comment, logs)
    if comment in [WARNING, FAIL]:
        with open(LOG_PATH, 'a') as log_f:
            log_f.write(str_log)
    if comment in [START, FINISH]:
        if pid % 500 != 0:
            return
    print(str_log, end='')


def cut_img(img, seq_info, frame_name, pid):
    # A silhouette contains too little white pixels
    # might be not valid for identification.
    if img.sum() <= 10000:
        message = 'seq:%s, frame:%s, no data, %d.' % (
            '-'.join(seq_info), frame_name, img.sum())
        warn(message)
        log_print(pid, WARNING, message)
        return None
    # Get the top and bottom point
    y = img.sum(axis=1)
    y_top = (y != 0).argmax(axis=0)
    y_btm = (y != 0).cumsum(axis=0).argmax(axis=0)
    img = img[y_top:y_btm + 1, :]
    # As the height of a person is larger than the width,
    # use the height to calculate resize ratio.
    _r = img.shape[1] / img.shape[0]
    _t_w = int(T_H * _r)
    img = cv2.resize(img, (_t_w, T_H), interpolation=cv2.INTER_CUBIC)
    # Get the median of x axis and regard it as the x center of the person.
    sum_point = img.sum()
    sum_column = img.sum(axis=0).cumsum()
    x_center = -1
    for i in range(sum_column.size):
        if sum_column[i] > sum_point / 2:
            x_center = i
            break
    if x_center < 0:
        message = 'seq:%s, frame:%s, no center.' % (
            '-'.join(seq_info), frame_name)
        warn(message)
        log_print(pid, WARNING, message)
        return None
    h_T_W = int(T_W / 2)
    left = x_center - h_T_W
    right = x_center + h_T_W
    if left <= 0 or right >= img.shape[1]:
        left += h_T_W
        right += h_T_W
        _ = np.zeros((img.shape[0], h_T_W))
        img = np.concatenate([_, img, _], axis=1)
    img = img[:, left:right]
    return img.astype('uint8')


def cut_pickle(seq_info, seq_info_origin, pid, _first_frame, _last_frame):
    seq_name = '-'.join(seq_info)
    log_print(pid, START, seq_name)
    seq_path = os.path.join(INPUT_PATH, *seq_info_origin)
    out_dir = os.path.join(OUTPUT_PATH, *seq_info)
    count_frame = 0
    for _frame_name in range(int(_first_frame), int(_last_frame)+1):
        _frame_name = str(_frame_name).zfill(8) + '.png'
        frame_path = os.path.join(seq_path, _frame_name)
        img = cv2.imread(frame_path)
        if img is not None:
            img = img[:, :, 0]
            img = cut_img(img, seq_info, _frame_name, pid)
            # Save the cut img
            save_path = os.path.join(out_dir, _frame_name)
            if os.path.exists(save_path):
                print(f'File : {save_path} exists.')
                count_frame += 1
                continue
            imageio.imwrite(save_path, img)
            count_frame += 1
        else:
            print(f'Image : {frame_path} is None.')
    # Warn if the sequence contains less than 5 frames
    if count_frame < 5:
        message = 'seq:%s, less than 5 valid data.' % (
            '-'.join(seq_info))
        warn(message)
        log_print(pid, WARNING, message)

    log_print(pid, FINISH,
              'Contain %d valid frames. Saved to %s.'
              % (count_frame, out_dir))


pool = Pool(WORKERS)
results = list()
pid = 0

print('Pretreatment Start.\n'
      'Input path: %s\n'
      'Output path: %s\n'
      'Log file: %s\n'
      'Worker num: %d' % (
          INPUT_PATH, OUTPUT_PATH, LOG_PATH, WORKERS))

IDList_dir = osp.join(opt.input_path, "OULP-C1V1_SubjectIDList(FormatVersion1.0)")
IDLists_g = [
    'IDList_OULP-C1V1-A-55_Gallery.csv',
    'IDList_OULP-C1V1-A-65_Gallery.csv',
    'IDList_OULP-C1V1-A-75_Gallery.csv',
    'IDList_OULP-C1V1-A-85_Gallery.csv',
]
IDLists_p = [
    'IDList_OULP-C1V1-A-55_Probe.csv',
    'IDList_OULP-C1V1-A-65_Probe.csv',
    'IDList_OULP-C1V1-A-75_Probe.csv',
    'IDList_OULP-C1V1-A-85_Probe.csv',
]

_seq_type = 'Seq00'
for _view_g in IDLists_g:
    _view = _view_g.split('_')[1].split('-')[-1]
    csvlines = open(os.path.join(IDList_dir, _view_g)).readlines()
    for line in csvlines:
        line = line.strip()
        _pid = line.split(',')[0].zfill(7)
        _first_frame = line.split(',')[2]
        _last_frame = line.split(',')[3]
        seq_info_origin = [_seq_type, line.split(',')[0]]
        seq_info_new = [_pid, _seq_type, _view]
        out_dir = os.path.join(OUTPUT_PATH, *seq_info_new)
        if not os.path.exists(out_dir):
            print(f'Create Output Dir : {out_dir}.')
            os.makedirs(out_dir)
        results.append(
            pool.apply_async(
                cut_pickle,
                args=(seq_info_new, seq_info_origin, int(_pid), _first_frame, _last_frame)))
        sleep(0.02)

_seq_type = 'Seq01'
for _view_p in IDLists_p:
    _view = _view_p.split('_')[1].split('-')[-1]
    csvlines = open(os.path.join(IDList_dir, _view_p)).readlines()
    for line in csvlines:
        line = line.strip()
        _pid = line.split(',')[0].zfill(7)
        _first_frame = line.split(',')[2]
        _last_frame = line.split(',')[3]
        seq_info_origin = [_seq_type, line.split(',')[0]]
        seq_info_new = [_pid, _seq_type, _view]
        out_dir = os.path.join(OUTPUT_PATH, *seq_info_new)
        if not os.path.exists(out_dir):
            print(f'Create Output Dir : {out_dir}.')
            os.makedirs(out_dir)
        results.append(
            pool.apply_async(
                cut_pickle,
                args=(seq_info_new, seq_info_origin, int(_pid), _first_frame, _last_frame)))
        sleep(0.02)


pool.close()
unfinish = 1
while unfinish > 0:
    unfinish = 0
    for i, res in enumerate(results):
        try:
            res.get(timeout=0.1)
        except Exception as e:
            if type(e) == MP_TimeoutError:
                unfinish += 1
                continue
            else:
                print(f'\n\n\nERROR OCCUR: PID ##{i}##, ERRORTYPE: {type(e)}\n\n\n')
                raise e
pool.join()
