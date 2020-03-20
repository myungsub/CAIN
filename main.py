import os
import sys
import time
import copy
import shutil
import random

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import config
import utils
from loss import Loss


##### Parse CmdLine Arguments #####
args, unparsed = config.get_args()
cwd = os.getcwd()
print(args)


##### TensorBoard & Misc Setup #####
if args.mode != 'test':
    writer = SummaryWriter('logs/%s' % args.exp_name)

device = torch.device('cuda' if args.cuda else 'cpu')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)


##### Load Dataset #####
train_loader, test_loader = utils.load_dataset(
    args.dataset, args.data_root, args.batch_size, args.test_batch_size, args.num_workers, args.test_mode)


##### Build Model #####
if args.model.lower() == 'cain_encdec':
    from model.cain_encdec import CAIN_EncDec
    print('Building model: CAIN_EncDec')
    model = CAIN_EncDec(depth=args.depth, start_filts=32)
elif args.model.lower() == 'cain':
    from model.cain import CAIN
    print("Building model: CAIN")
    model = CAIN(depth=args.depth)
elif args.model.lower() == 'cain_noca':
    from model.cain_noca import CAIN_NoCA
    print("Building model: CAIN_NoCA")
    model = CAIN_NoCA(depth=args.depth)
else:
    raise NotImplementedError("Unknown model!")
# Just make every model to DataParallel
model = torch.nn.DataParallel(model).to(device)
#print(model)

##### Define Loss & Optimizer #####
criterion = Loss(args)

args.radam = False
if args.radam:
    from radam import RAdam
    optimizer = RAdam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
else:
    from torch.optim import Adam
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
print('# of parameters: %d' % sum(p.numel() for p in model.parameters()))


# If resume, load checkpoint: model + optimizer
if args.resume:
    utils.load_checkpoint(args, model, optimizer)

# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True)


# Initialize LPIPS model if used for evaluation
# lpips_model = utils.init_lpips_eval() if args.lpips else None
lpips_model = None

LOSS_0 = 0


def train(args, epoch):
    global LOSS_0
    losses, psnrs, ssims, lpips = utils.init_meters(args.loss)
    model.train()
    criterion.train()

    t = time.time()
    for i, (images, imgpaths) in enumerate(train_loader):

        # Build input batch
        im1, im2, gt = utils.build_input(images, imgpaths)

        # Forward
        optimizer.zero_grad()
        out, feats = model(im1, im2)
        loss, loss_specific = criterion(out, gt, None, feats)

        # Save loss values
        for k, v in losses.items():
            if k != 'total':
                v.update(loss_specific[k].item())
        if LOSS_0 == 0:
            LOSS_0 = loss.data.item()
        losses['total'].update(loss.item())

        # Backward (+ grad clip) - if loss explodes, skip current iteration
        loss.backward()
        if loss.data.item() > 10.0 * LOSS_0:
            print(max(p.grad.data.abs().max() for p in model.parameters()))
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        # Calc metrics & print logs
        if i % args.log_iter == 0:
            utils.eval_metrics(out, gt, psnrs, ssims, lpips, lpips_model)

            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}\tPSNR: {:.4f}\tTime({:.2f})'.format(
                epoch, i, len(train_loader), losses['total'].avg, psnrs.avg, time.time() - t))
            
            # Log to TensorBoard
            utils.log_tensorboard(writer, losses, psnrs.avg, ssims.avg, lpips.avg,
                optimizer.param_groups[-1]['lr'], epoch * len(train_loader) + i)

            # Reset metrics
            losses, psnrs, ssims, lpips = utils.init_meters(args.loss)
            t = time.time()


def test(args, epoch, eval_alpha=0.5):
    print('Evaluating for epoch = %d' % epoch)
    losses, psnrs, ssims, lpips = utils.init_meters(args.loss)
    model.eval()
    criterion.eval()

    save_folder = 'test%03d' % epoch
    if args.dataset == 'snufilm':
        save_folder = os.path.join(save_folder, args.dataset, args.test_mode)
    else:
        save_folder = os.path.join(save_folder, args.dataset)
    save_dir = os.path.join('checkpoint', args.exp_name, save_folder)
    utils.makedirs(save_dir)
    save_fn = os.path.join(save_dir, 'results.txt')
    if not os.path.exists(save_fn):
        with open(save_fn, 'w') as f:
            f.write('For epoch=%d\n' % epoch)

    t = time.time()
    with torch.no_grad():
        for i, (images, imgpaths) in enumerate(tqdm(test_loader)):

            # Build input batch
            im1, im2, gt = utils.build_input(images, imgpaths, is_training=False)

            # Forward
            out, feats = model(im1, im2)

            # Save loss values
            loss, loss_specific = criterion(out, gt, None, feats)
            for k, v in losses.items():
                if k != 'total':
                    v.update(loss_specific[k].item())
            losses['total'].update(loss.item())

            # Evaluate metrics
            utils.eval_metrics(out, gt, psnrs, ssims, lpips)

            # Log examples that have bad performance
            if (ssims.val < 0.9 or psnrs.val < 25) and epoch > 50:
                print(imgpaths)
                print("\nLoss: %f, PSNR: %f, SSIM: %f, LPIPS: %f" %
                      (losses['total'].val, psnrs.val, ssims.val, lpips.val))
                print(imgpaths[1][-1])

            # Save result images
            if ((epoch + 1) % 1 == 0 and i < 20) or args.mode == 'test':
                savepath = os.path.join('checkpoint', args.exp_name, save_folder)

                for b in range(images[0].size(0)):
                    paths = imgpaths[1][b].split('/')
                    fp = os.path.join(savepath, paths[-3], paths[-2])
                    if not os.path.exists(fp):
                        os.makedirs(fp)
                    # remove '.png' extension
                    fp = os.path.join(fp, paths[-1][:-4])
                    utils.save_image(out[b], "%s.png" % fp)
                    
    # Print progress
    print('im_processed: {:d}/{:d} {:.3f}s   \r'.format(i + 1, len(test_loader), time.time() - t))
    print("Loss: %f, PSNR: %f, SSIM: %f, LPIPS: %f\n" %
          (losses['total'].avg, psnrs.avg, ssims.avg, lpips.avg))

    # Save psnr & ssim
    save_fn = os.path.join('checkpoint', args.exp_name, save_folder, 'results.txt')
    with open(save_fn, 'a') as f:
        f.write("PSNR: %f, SSIM: %f, LPIPS: %f\n" %
                (psnrs.avg, ssims.avg, lpips.avg))

    # Log to TensorBoard
    if args.mode != 'test':
        utils.log_tensorboard(writer, losses, psnrs.avg, ssims.avg, lpips.avg,
            optimizer.param_groups[-1]['lr'], epoch * len(train_loader) + i, mode='test')

    return losses['total'].avg, psnrs.avg, ssims.avg, lpips.avg


""" Entry Point """
def main(args):
    if args.mode == 'test':
        _, _, _, _ = test(args, args.start_epoch)
        return

    best_psnr = 0
    for epoch in range(args.start_epoch, args.max_epoch):
        
        # run training
        train(args, epoch)

        # run test
        test_loss, psnr, _, _ = test(args, epoch)

        # save checkpoint
        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)
        utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_psnr': best_psnr
        }, is_best, args.exp_name)

        # update optimizer policy
        scheduler.step(test_loss)

if __name__ == "__main__":
    main(args)
