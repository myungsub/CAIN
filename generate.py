import os
import sys
import time
import copy
import shutil
import random

import torch
import numpy as np
from tqdm import tqdm

import config
import utils


##### Parse CmdLine Arguments #####
args, unparsed = config.get_args()
cwd = os.getcwd()
print(args)


device = torch.device('cuda' if args.cuda else 'cpu')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)




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

print('# of parameters: %d' % sum(p.numel() for p in model.parameters()))


# If resume, load checkpoint: model
if args.resume:
    #utils.load_checkpoint(args, model, optimizer=None)
    checkpoint = torch.load('pretrained_cain.pth')
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['state_dict'])
    del checkpoint



def test(args, epoch):
    print('Evaluating for epoch = %d' % epoch)
    ##### Load Dataset #####
    test_loader = utils.load_dataset(
        args.dataset, args.data_root, args.batch_size, args.test_batch_size, args.num_workers, img_fmt=args.img_fmt)
    model.eval()

    t = time.time()
    with torch.no_grad():
        for i, (images, meta) in enumerate(tqdm(test_loader)):

            # Build input batch
            im1, im2 = images[0].to(device), images[1].to(device)

            # Forward
            out, _ = model(im1, im2)

            # Save result images
            if args.mode == 'test':
                for b in range(images[0].size(0)):
                    paths = meta['imgpath'][0][b].split('/')
                    fp = args.data_root
                    fp = os.path.join(fp, paths[-1][:-4])   # remove '.png' extension
                    
                    # Decide float index
                    i1_str = paths[-1][:-4]
                    i2_str = meta['imgpath'][1][b].split('/')[-1][:-4]
                    try:
                        i1 = float(i1_str.split('_')[-1])
                    except ValueError:
                        i1 = 0.0
                    try:
                        i2 = float(i2_str.split('_')[-1])
                        if i2 == 0.0:
                            i2 = 1.0
                    except ValueError:
                        i2 = 1.0
                    fpos = max(0, fp.rfind('_'))
                    fInd = (i1 + i2) / 2
                    savepath = "%s_%06f.%s" % (fp[:fpos], fInd, args.img_fmt)
                    utils.save_image(out[b], savepath)
                    
    # Print progress
    print('im_processed: {:d}/{:d} {:.3f}s   \r'.format(i + 1, len(test_loader), time.time() - t))

    return


""" Entry Point """
def main(args):

    num_iter = 2 # x2**num_iter interpolation
    for _ in range(num_iter):
        
        # run test
        test(args, args.start_epoch)


if __name__ == "__main__":
    main(args)
