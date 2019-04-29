import os as _os
import glob as _glob

BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))
SCRATCH_DIR = _os.path.join(BASE_DIR, 'scratch')
LOGS_DIR = _os.path.join(SCRATCH_DIR, 'logs')
TFRECORDS_DIR = _os.path.join(SCRATCH_DIR, 'data')
MODEL_CKPT_DIR = _os.path.join(SCRATCH_DIR, 'checkpoints')

TFRECORDS_FILENAMES = _glob.glob(_os.path.join(
    TFRECORDS_DIR, 'validation-0000[012]-of-00128'))  # 1171 imgs

MODEL_NAME_TO_CKPT_PATH_MAP = dict()
MODEL_NAME_TO_CKPT_PATH_MAP.update({
    'new-%d' % q: _os.path.join(
        MODEL_CKPT_DIR, 'resnet-keras-jpeg%d-dgx1.h5' % q) 
    for q in [80, 60, 40, 20]})
MODEL_NAME_TO_CKPT_PATH_MAP.update({
    'old-%d' % q: _os.path.join(
        MODEL_CKPT_DIR, 'resnet_50_v2-jpeg_%d' % q) 
    for q in [80, 60, 40, 20]})
