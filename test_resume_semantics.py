"""Checks the two PyTorch behaviours that train.py's resume path relies on.

Run it anywhere (CPU is enough; on a CUDA machine it also checks the device
placement of optimizer state):

    python test_resume_semantics.py

It uses a tiny stand-in model, so it takes a second and needs no dataset. It
mirrors how train.py builds / saves / resumes the optimizer and scheduler:

  * save happens BEFORE scheduler.step()  (as in train_net)
  * resume builds the model on the target device first, then the optimizer,
    then optimizer.load_state_dict(), then
    MultiStepLR(..., last_epoch=start_epoch - 1)

1. LR trace of an interrupted-and-resumed run must equal the uninterrupted one
   for every possible interruption point.
2. After a resume, every optimizer state tensor must live on the same device
   as its parameter.

The 'old behaviour' control (last_epoch left at -1) is expected to DIVERGE; it
is printed so you can see that the test is actually sensitive to the bug.
"""
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR

MILESTONES = [3, 6, 9, 12]
TOTAL_EPOCHS = 15
GAMMA = 0.1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build(snapshot=None, start_epoch=0, fix_last_epoch=True):
    """Build model/optimizer/scheduler exactly in train.py's order."""
    torch.manual_seed(0)
    model = nn.Linear(4, 2).to(DEVICE)          # device first ...
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)  # ... then optimizer
    if snapshot is not None:
        model.load_state_dict(snapshot['model'])
        opt.load_state_dict(snapshot['opt'])
    last_epoch = start_epoch - 1 if fix_last_epoch else -1
    sched = MultiStepLR(opt, milestones=MILESTONES, gamma=GAMMA, last_epoch=last_epoch)
    return model, opt, sched


def run(model, opt, sched, first_epoch, stop_after=None):
    """Train toy epochs; return (lr per epoch, snapshot saved after `stop_after`)."""
    lrs, snap = [], None
    for epoch in range(first_epoch, TOTAL_EPOCHS):
        lrs.append(opt.param_groups[0]['lr'])
        x = torch.randn(8, 4, device=DEVICE)
        loss = model(x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch == stop_after:                  # save_checkpoint() happens here ...
            snap = {'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    'opt': opt.state_dict()}
            snap = torch.load(_roundtrip(snap), weights_only=True, map_location='cpu')
            return lrs, snap
        sched.step()                             # ... and the scheduler steps afterwards
    return lrs, snap


def _roundtrip(obj):
    import io
    buf = io.BytesIO()
    torch.save(obj, buf)
    buf.seek(0)
    return buf


def main():
    reference, _ = run(*build(), first_epoch=0)
    print('uninterrupted LR trace:', ['%.0e' % v for v in reference])

    bad_fixed, bad_old = [], []
    for stop in range(0, TOTAL_EPOCHS - 1):
        pre, snap = run(*build(), first_epoch=0, stop_after=stop)
        start = stop + 1

        model, opt, sched = build(snap, start_epoch=start, fix_last_epoch=True)
        # optimizer state must sit on the parameters' device
        for p in model.parameters():
            for t in opt.state[p].values():
                if torch.is_tensor(t):
                    assert t.device == p.device, 'optimizer state on {} but param on {}'.format(t.device, p.device)
        post, _ = run(model, opt, sched, first_epoch=start)
        if pre + post != reference:
            bad_fixed.append(stop)

        model, opt, sched = build(snap, start_epoch=start, fix_last_epoch=False)   # the old behaviour
        post_old, _ = run(model, opt, sched, first_epoch=start)
        if pre + post_old != reference:
            bad_old.append(stop)

    print('device used:', DEVICE)
    print('resume points where the FIXED order diverges :', bad_fixed or 'none')
    print('resume points where the OLD order diverges   :', bad_old or 'none (test is not sensitive!)')
    assert not bad_fixed, 'resume changes the LR schedule'
    assert bad_old, 'control did not diverge: this test would not catch the original bug'
    print('RESUME SEMANTICS OK')


if __name__ == '__main__':
    main()
