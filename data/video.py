import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class Video(Dataset):
    def __init__(self, data_root, fmt='png'):
        images = sorted(glob.glob(os.path.join(data_root, '*.%s' % fmt)))
        for im in images:
            try:
                float_ind = float(im.split('_')[-1][:-4])
            except ValueError:
                os.rename(im, '%s_%.06f.%s' % (im[:-4], 0.0, fmt))
        # re
        images = sorted(glob.glob(os.path.join(data_root, '*.%s' % fmt)))
        self.imglist = [[images[i], images[i+1]] for i in range(len(images)-1)]
        print('[%d] images ready to be loaded' % len(self.imglist))


    def __getitem__(self, index):
        imgpaths = self.imglist[index]

        # Load images
        img1 = Image.open(imgpaths[0])
        img2 = Image.open(imgpaths[1])

        T = transforms.ToTensor()
        img1 = T(img1)
        img2 = T(img2)

        imgs = [img1, img2] 
        meta = {'imgpath': imgpaths}
        return imgs, meta

    def __len__(self):
        return len(self.imglist)


def get_loader(mode, data_root, batch_size, img_fmt='png', shuffle=False, num_workers=0, n_frames=1):
    if mode == 'train':
        is_training = True
    else:
        is_training = False
    dataset = Video(data_root, fmt=img_fmt)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
