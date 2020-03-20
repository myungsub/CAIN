import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class SNUFILM(Dataset):
    def __init__(self, data_root, mode='hard'):
        '''
        :param data_root:   ./data/SNU-FILM
        :param mode:        ['easy', 'medium', 'hard', 'extreme']
        '''
        test_root = os.path.join(data_root, 'test')
        test_fn = os.path.join(data_root, 'test-%s.txt' % mode)
        with open(test_fn, 'r') as f:
            self.frame_list = f.read().splitlines()
        self.frame_list = [v.split(' ') for v in self.frame_list]
        
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        
        print("[%s] Test dataset has %d triplets" %  (mode, len(self.frame_list)))


    def __getitem__(self, index):
        
        # Use self.test_all_images:
        imgpaths = self.frame_list[index]
        
        img1 = Image.open(imgpaths[0])
        img2 = Image.open(imgpaths[1])
        img3 = Image.open(imgpaths[2])

        img1 = self.transforms(img1)
        img2 = self.transforms(img2)
        img3 = self.transforms(img3)

        imgs = [img1, img2, img3]

        return imgs, imgpaths

    def __len__(self):
        return len(self.frame_list)


def check_already_extracted(vid):
    return bool(os.path.exists(vid + '/0001.png'))


def get_loader(mode, data_root, batch_size, shuffle, num_workers, test_mode='hard'):
    # data_root = 'data/SNUFILM'
    dataset = SNUFILM(data_root, mode=test_mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
