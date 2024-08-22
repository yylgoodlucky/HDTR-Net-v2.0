import cv2, os
import torch
import random
import numpy as np
import torch.utils.data as data
from os.path import join
from tqdm import tqdm

from utils.data_util import *
from torchvision.transforms.functional import normalize

class Datagan_vqgan(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.resize = self.opt['dataset'].get('resize')
        self.filelist = self.get_filelists()
        
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        
    def get_filelists(self):
        """ filelist include file path.
            example: /user/vfhq/group/file.
        """
        filelist = []
        with open(self.opt['filelist'], 'r') as f:
            print(f'info : read data ...')
            for line in tqdm(f):
                line = line.strip()
                if ' ' in line: line = line.split()[0]
                for imgfile in os.listdir(self.opt['dpath'] + line):
                    filelist.append(join(self.opt['dpath'] + line, imgfile))

        return filelist
    
    def crop_mouth(self, img: np.array, lmark: np.array):
        """ according landmarks crop mouth.
            tip of nose index : 54
            down of jaw index : 15
            left of mouth index : 76
            right of mouth index : 82
        """
        lmark = lmark.astype(int)
        # define index
        up, down = lmark[54][1], lmark[15][1]
        left, right = lmark[76][0], lmark[82][0]
        
        # padding to square
        pad = ((down - up) - (right - left)) / 2
        left = int(left - pad)
        right = int(right + pad)
        
        return img[up: down, left: right, :]
        
    def __getitem__(self, idx):
        # select img
        img_path = self.filelist[idx]
        lmark_path = img_path.replace('.png', '.npy').replace('align_images', 'landmark98')
        
        # load img, lmark
        if os.path.exists(lmark_path) and os.path.exists(img_path):
            lmark = np.load(lmark_path)
            image = cv2.imread(img_path)
        else:
            return self.__getitem__(np.random.randint(0, len(self.filelist)))

        # according landmark crop mouth
        image = self.crop_mouth(image, lmark)
        if image.shape[0] < 100 or image.shape[1] < 100:
            return self.__getitem__(np.random.randint(0, len(self.filelist)))
        
        image = cv2.resize(image, (self.resize, self.resize))
        img_in = img2tensor(image, bgr2rgb=True, float32=True)
        img_in = np.clip((img_in * 255.0).round(), 0, 255) / 255.

        normalize(img_in, self.mean, self.std, inplace=True)
        
        return {'in': img_in}
    
    def __len__(self):
        return len(self.filelist)