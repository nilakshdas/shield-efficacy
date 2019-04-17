import os as _os
import glob as _glob

BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))
SCRATCH_DIR = _os.path.join(BASE_DIR, 'scratch')
LOGS_DIR = _os.path.join(SCRATCH_DIR, 'logs')
TFRECORDS_DIR = _os.path.join(SCRATCH_DIR, 'data')
MODEL_CKPT_DIR = _os.path.join(SCRATCH_DIR, 'checkpoints')

TFRECORDS_FILENAMES = _glob.glob(_os.path.join(
    TFRECORDS_DIR, 'validation-0000[012]-of-00128'))  # 1171 imgs

MODEL_CKPT_PATH_FORMAT = _os.path.join(
    MODEL_CKPT_DIR, 'resnet-keras-jpeg{q}-dgx1.h5')
