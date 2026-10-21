import os

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Model parameters
image_w = 112
image_h = 112
channel = 3
emb_size = 128

# ---------------------------------------------------------------------------
# Input normalization -- SINGLE source of truth for train / eval / export.
#
# 'esp'      : (x - 127.5) / 127.5, i.e. Normalize(0.5, 0.5) after ToTensor.
#              Use this when training from scratch for ESP-DL / ESP-PPQ
#              deployment: the official Espressif face pipeline feeds the
#              model with mean/std 127.5, so training, calibration and
#              firmware preprocessing all agree.
# 'imagenet' : ImageNet mean/std. This is what foamliu's original training
#              used. You MUST select it when evaluating the released
#              pretrained weights, otherwise LFW numbers will be off.
# ---------------------------------------------------------------------------
NORM_PRESET = 'esp'
_NORM_PRESETS = {
    'esp': ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    'imagenet': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}
NORM_MEAN, NORM_STD = _NORM_PRESETS[NORM_PRESET]

# Training parameters
num_workers = 4  # for data-loading; right now, only 1 works with h5py
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none

# Data parameters
# faces_emore (refined MS-Celeb-1M, "emore") full set -- author's final values.
# NOTE: upstream master had num_classes=2964 / num_samples=200000 here, left
# over from a contributor who trained on only the first 200k images. Those
# numbers match no published dataset and silently truncate training.
num_classes = 85742
num_samples = 5822653
DATA_DIR = 'data'
faces_ms1m_folder = 'data/faces_emore'
path_imgidx = os.path.join(faces_ms1m_folder, 'train.idx')
path_imgrec = os.path.join(faces_ms1m_folder, 'train.rec')
IMG_DIR = 'data/images'
pickle_file = 'data/faces_ms1m_112x112.pickle'
pretrained_model_path = "pretrained_model/mobilefacenet_scripted.pt"
