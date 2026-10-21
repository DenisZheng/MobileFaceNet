import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from config import device, grad_clip, print_freq
from data_gen import ArcFaceDataset
from focal_loss import FocalLoss
from lfw_eval import lfw_test
from mobilefacenet import MobileFaceNet, ArcMarginModel
from facenet_utils import parse_args, save_checkpoint, load_checkpoint_file, AverageMeter, accuracy, get_logger, clip_gradient


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_acc = float('-inf')
    writer = SummaryWriter()
    epochs_since_improvement = 0

    # Initialize model / metric head / optimizer (always built the same
    # way; a resume then loads state_dicts into them). Upstream instead
    # unpickled whole module objects out of the checkpoint, which modern
    # PyTorch refuses to load (weights_only) and which baked in the
    # original author's class paths and DataParallel wrapper.
    model = MobileFaceNet()
    metric_fc = ArcMarginModel(args)

    optimizer = torch.optim.SGD([{'params': model.conv1.parameters()},
                                 {'params': model.dw_conv.parameters()},
                                 {'params': model.features.parameters()},
                                 {'params': model.conv2.parameters()},
                                 {'params': model.gdconv.parameters()},
                                 {'params': model.conv3.parameters(), 'weight_decay': 4e-4},
                                 {'params': model.bn.parameters()},
                                 {'params': metric_fc.parameters()}],
                                lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay, nesterov=True)

    if checkpoint is not None:
        ckpt = load_checkpoint_file(checkpoint)
        if 'model_state' in ckpt:
            start_epoch = ckpt['epoch'] + 1
            epochs_since_improvement = ckpt['epochs_since_improvement']
            best_acc = ckpt.get('acc', best_acc)
            model.load_state_dict(ckpt['model_state'])
            metric_fc.load_state_dict(ckpt['metric_fc_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
        else:
            raise ValueError(
                '{} is a legacy whole-object checkpoint; convert it first '
                'with export.py (it understands the legacy format), then '
                'resume from a state_dict checkpoint.'.format(checkpoint))

    # Only wrap for genuine multi-GPU; DataParallel on a single GPU/CPU
    # just adds overhead and 'module.' prefixes to every state_dict key.
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        metric_fc = nn.DataParallel(metric_fc)

    logger = get_logger()

    # Move to GPU, if available
    model = model.to(device)
    metric_fc = metric_fc.to(device)

    # Loss function
    if args.focal_loss:
        criterion = FocalLoss(gamma=args.gamma).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    train_dataset = ArcFaceDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Author's milestones; upstream master had [5, 10, 15, 20] from the
    # same small-subset run that produced the lr=1e-4 default.
    # NOTE on resume: the scheduler restarts its count, so a resumed run
    # re-arms these milestones -- acceptable for this codebase's scale,
    # but don't expect bit-exact continuation of the LR schedule.
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.1)

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        # One epoch's training
        train_loss, train_acc = train(train_loader=train_loader,
                                      model=model,
                                      metric_fc=metric_fc,
                                      criterion=criterion,
                                      optimizer=optimizer,
                                      epoch=epoch,
                                      logger=logger)

        lr = optimizer.param_groups[0]['lr']
        print('\nLearning rate={}\n'.format(lr))

        writer.add_scalar('model/train_loss', train_loss, epoch)
        writer.add_scalar('model/train_acc', train_acc, epoch)
        writer.add_scalar('model/learning_rate', lr, epoch)

        # One epoch's validation
        lfw_acc, threshold = lfw_test(model)
        writer.add_scalar('model/lfw_acc', lfw_acc, epoch)
        writer.add_scalar('model/threshold', threshold, epoch)

        # Check if there was an improvement
        is_best = lfw_acc > best_acc
        best_acc = max(lfw_acc, best_acc)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, metric_fc, optimizer, best_acc, is_best)
        scheduler.step()


def train(train_loader, model, metric_fc, criterion, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)
    metric_fc.train()

    losses = AverageMeter()
    top5_accs = AverageMeter()

    # Batches
    for i, (img, label) in enumerate(train_loader):
        # Move to GPU, if available
        img = img.to(device)
        label = label.to(device)  # [N, 1]

        # Forward prop.
        feature = model(img)  # embedding => [N, 128]
        output = metric_fc(feature, label)  # class_id_out => [N, num_classes]

        # Calculate loss
        loss = criterion(output, label)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())
        top5_accuracy = accuracy(output, label, 5)
        top5_accs.update(top5_accuracy)

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Top5 Accuracy {top5_accs.val:.3f} ({top5_accs.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                                         loss=losses,
                                                                                         top5_accs=top5_accs))

    return losses.avg, top5_accs.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
