import time

import torch

from facenet_utils import load_model_from_checkpoint

if __name__ == '__main__':
    # Accepts new state_dict checkpoints, legacy whole-object checkpoints
    # (e.g. an original BEST_checkpoint.tar), or a bare state_dict.
    checkpoint_path = 'BEST_checkpoint.tar'
    print('loading {}...'.format(checkpoint_path))
    start = time.time()
    model = load_model_from_checkpoint(checkpoint_path, torch.device('cpu'))
    print('elapsed {} sec'.format(time.time() - start))
    print(type(model))

    filename = 'mobilefacenet.pt'
    print('saving {}...'.format(filename))
    start = time.time()
    torch.save(model.state_dict(), filename)
    print('elapsed {} sec'.format(time.time() - start))

    scripted_model_file = 'mobilefacenet_scripted.pt'
    print('saving {}...'.format(scripted_model_file))
    torch.jit.save(torch.jit.script(model), scripted_model_file)
    print('done. For ESP-DL deployment, run export_onnx.py next.')
