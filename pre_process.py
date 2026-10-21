import os
import pickle
import struct

import cv2 as cv
import numpy as np
from tqdm import tqdm

from config import path_imgidx, path_imgrec, IMG_DIR, pickle_file, num_samples
from facenet_utils import ensure_folder

# ---------------------------------------------------------------------------
# Reads faces_emore train.rec/train.idx and unpacks it into IMG_DIR +
# pickle_file for data_gen.ArcFaceDataset.
#
# Upstream master hardcoded range(200000) here (the same contributor
# subset run that left num_samples=200000 in config.py). Restored to
# config.num_samples, i.e. the full dataset; the loop stops cleanly at
# the end of the record file either way.
#
# mxnet is used when importable, but it is effectively unmaintained and
# has no wheels for recent Python versions, so a small built-in RecordIO
# reader is provided as fallback (same on-disk format: magic 0xced7230a,
# '<IfQQ' header, JPEG payload).
# ---------------------------------------------------------------------------

_REC_MAGIC = 0xced7230a
_IR_FORMAT = 'IfQQ'
_IR_SIZE = struct.calcsize(_IR_FORMAT)


def _iter_records_mxnet():
    from mxnet import recordio
    import mxnet as mx
    imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
    for i in range(num_samples):
        try:
            header, s = recordio.unpack(imgrec.read_idx(i + 1))
        except Exception:
            break
        img = mx.image.imdecode(s).asnumpy()          # RGB
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)      # -> BGR for imwrite
        yield int(header.label), img


def _read_record_at(rec_file, start, end):
    rec_file.seek(start)
    chunk = rec_file.read(end - start)
    pieces = []
    offset = 0
    while offset < len(chunk):
        magic, lrecord = struct.unpack_from('<II', chunk, offset)
        if magic != _REC_MAGIC:
            raise ValueError('bad RecordIO magic')
        cflag = (lrecord >> 29) & 0x7
        length = lrecord & ((1 << 29) - 1)
        offset += 8
        pieces.append(chunk[offset:offset + length])
        offset += length + (-(8 + length)) % 4
        if cflag in (0, 3):
            break
    return b''.join(pieces)


def _iter_records_fallback():
    index = []
    with open(path_imgidx, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                key, pos = line.split('\t')
                index.append((int(key), int(pos)))
    index.sort(key=lambda kv: kv[1])
    positions = dict(index)
    keys = [k for k, _ in index]
    with open(path_imgrec, 'rb') as rec_file:
        rec_file.seek(0, os.SEEK_END)
        file_size = rec_file.tell()
        for n, key in enumerate(keys):
            if n == 0:
                continue  # record 0 holds dataset meta, not an image
            if n >= num_samples:
                break
            start = positions[key]
            end = positions[keys[n + 1]] if n + 1 < len(keys) else file_size
            try:
                record = _read_record_at(rec_file, start, end)
                flag, label, _id, _id2 = struct.unpack_from(_IR_FORMAT, record, 0)
                payload = record[_IR_SIZE:]
                if flag > 0:
                    label = float(np.frombuffer(payload[:flag * 4], dtype=np.float32)[0])
                    payload = payload[flag * 4:]
                img = cv.imdecode(np.frombuffer(payload, dtype=np.uint8), cv.IMREAD_COLOR)  # BGR
                if img is None:
                    continue
            except Exception:
                continue
            yield int(label), img


def iter_records():
    try:
        import mxnet  # noqa: F401
        print('using mxnet RecordIO reader')
        yield from _iter_records_mxnet()
    except ImportError:
        print('mxnet not available, using built-in RecordIO reader')
        yield from _iter_records_fallback()


if __name__ == "__main__":
    ensure_folder(IMG_DIR)

    samples = []
    class_ids = set()

    for i, (label, img) in enumerate(tqdm(iter_records())):
        class_ids.add(label)
        filename = '{}.jpg'.format(i)
        samples.append({'img': filename, 'label': label})
        cv.imwrite(os.path.join(IMG_DIR, filename), img)

    with open(pickle_file, 'wb') as file:
        pickle.dump(samples, file)

    print('num_samples: ' + str(len(samples)))
    print('num_classes: ' + str(len(class_ids)))
    if class_ids:
        print('max label: ' + str(max(class_ids)))
    if len(class_ids) != 85742:
        print('WARNING: expected 85742 classes for faces_emore; '
              'config.num_classes must match this number before training.')
