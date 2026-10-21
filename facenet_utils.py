import argparse
import logging

import cv2 as cv
import numpy as np
import torch
from PIL import Image

from align_faces import get_reference_facial_points, warp_and_crop_face
from config import image_h, image_w

_detector = None


def get_detector():
    """Lazily build the RetinaFace detector on first use.

    Upstream master imported (and instantiated) the detector at module
    import time, which loaded a CUDA model as a side effect of
    ``import facenet_utils`` -- so even train.py crashed on machines
    where RetinaFace/CUDA was unavailable, although training itself
    never uses the detector.
    """
    global _detector
    if _detector is None:
        from retinaface.detector import RetinafaceDetector
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        _detector = RetinafaceDetector(net='mnet', type=dev)
    return _detector


def _unwrap(module):
    return module.module if isinstance(module, torch.nn.DataParallel) else module


def save_checkpoint(epoch, epochs_since_improvement, model, metric_fc, optimizer, acc, is_best):
    """Save state_dicts, not whole pickled module objects.

    Upstream saved the live model/optimizer objects. That format cannot
    be loaded by modern PyTorch (torch.load defaults to weights_only=True
    since 2.6) and it bakes in DataParallel wrappers and class paths.
    """
    print('saving checkpoint ...')
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'acc': acc,
             'model_state': _unwrap(model).state_dict(),
             'metric_fc_state': _unwrap(metric_fc).state_dict(),
             'optimizer_state': optimizer.state_dict()}
    filename = 'checkpoint.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_checkpoint.tar')


def load_checkpoint_file(path):
    """Load a checkpoint saved by save_checkpoint, or a legacy upstream one.

    Legacy checkpoints contain pickled module objects, so they can only
    be read with weights_only=False. Only do that for files you trust
    (e.g. your own training output or the original repo's checkpoints).
    """
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        return torch.load(path, map_location='cpu', weights_only=False)


def _strip_module_prefix(state_dict):
    if any(k.startswith('module.') for k in state_dict):
        return {k[len('module.'):] if k.startswith('module.') else k: v
                for k, v in state_dict.items()}
    return state_dict


def load_model_from_checkpoint(path, device):
    """Build a MobileFaceNet and load backbone weights from a checkpoint.

    Accepts both the new state_dict format and legacy upstream
    checkpoints of the form {'model': <DataParallel module>, ...}.
    Returns the unwrapped model in eval mode on ``device``.
    """
    from mobilefacenet import MobileFaceNet
    checkpoint = load_checkpoint_file(path)
    model = MobileFaceNet()
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(_strip_module_prefix(checkpoint['model_state']))
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        legacy = _unwrap(checkpoint['model'])
        model.load_state_dict(_strip_module_prefix(legacy.state_dict()))
    else:  # a bare state_dict, e.g. mobilefacenet.pt from export.py
        model.load_state_dict(_strip_module_prefix(checkpoint))
    model = model.to(device)
    model.eval()
    return model


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k=1):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def align_face(img_fn, facial5points):
    raw = cv.imread(img_fn, 1)  # BGR
    facial5points = np.reshape(facial5points, (2, 5))

    crop_size = (image_h, image_w)

    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)
    output_size = (image_h, image_w)

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)

    # dst_img = warp_and_crop_face(raw, facial5points)
    dst_img = warp_and_crop_face(raw, facial5points, reference_pts=reference_5pts, crop_size=crop_size)
    return dst_img


def get_face_attributes(full_path):
    try:
        img = Image.open(full_path).convert('RGB')
        bounding_boxes, landmarks = get_detector().detect_faces(img)

        if len(landmarks) > 0:
            landmarks = [int(round(x)) for x in landmarks[0]]
            return True, landmarks

    except KeyboardInterrupt:
        raise
    except:
        pass
    return False, None


def select_significant_face(bounding_boxes):
    best_index = -1
    best_rank = float('-inf')
    for i, b in enumerate(bounding_boxes):
        bbox_w, bbox_h = b[2] - b[0], b[3] - b[1]
        area = bbox_w * bbox_h
        score = b[4]
        rank = score * area
        if rank > best_rank:
            best_rank = rank
            best_index = i

    return best_index


def get_central_face_attributes(full_path):
    try:
        img = cv.imread(full_path)
        bounding_boxes, landmarks = get_detector().detect_faces(img)

        if len(landmarks) > 0:
            i = select_significant_face(bounding_boxes)
            return True, [bounding_boxes[i]], [landmarks[i]]

    except KeyboardInterrupt:
        raise
    except ValueError:
        pass
    except IOError:
        pass
    return False, None, None


def get_all_face_attributes(full_path):
    img = Image.open(full_path).convert('RGB')
    bounding_boxes, landmarks = get_detector().detect_faces(img)
    return bounding_boxes, landmarks


def draw_bboxes(img, bounding_boxes, facial_landmarks=[]):
    for b in bounding_boxes:
        cv.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255), 1)

    for p in facial_landmarks:
        for i in range(5):
            cv.circle(img, (int(p[i]), int(p[i + 5])), 1, (0, 255, 0), -1)

        break  # only first

    return img


def str2bool(v):
    # argparse with type=bool is a trap: bool('False') is True, so
    # `--focal-loss False` used to ENABLE the option. Parse explicitly.
    if isinstance(v, bool):
        return v
    return v.lower() in ('1', 'true', 'yes', 'y')


def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--pretrained', type=str2bool, default=False, help='pretrained model')
    # Upstream master had end-epoch=30 / lr=0.0001 / batch-size=256, left
    # over from a contributor's small-subset run; with lr=1e-4 an ArcFace
    # model does not converge. Restored to the author's recipe values
    # (lr 0.1, batch 512); end-epoch 50 pairs with the [10,20,30,40]
    # LR milestones in train.py (upstream default was 1000 = "run until
    # stopped manually").
    parser.add_argument('--end-epoch', type=int, default=50, help='training epoch size.')
    parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
    parser.add_argument('--optimizer', default='sgd', help='optimizer')
    parser.add_argument('--weight-decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--emb-size', type=int, default=128, help='embedding length (informational; the backbone output is fixed at 128 by config.emb_size / conv3)')
    parser.add_argument('--batch-size', type=int, default=512, help='batch size in each context')
    parser.add_argument('--margin-m', type=float, default=0.5, help='angular margin m')
    parser.add_argument('--margin-s', type=float, default=64.0, help='feature scale s')
    parser.add_argument('--easy-margin', type=str2bool, default=False, help='easy margin')
    parser.add_argument('--focal-loss', type=str2bool, default=False, help='focal loss')
    parser.add_argument('--gamma', type=float, default=2.0, help='focusing parameter gamma')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    args = parser.parse_args()
    return args


def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def ensure_folder(folder):
    import os
    if not os.path.isdir(folder):
        os.mkdir(folder)
