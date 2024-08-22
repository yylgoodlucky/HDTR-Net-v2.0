import numpy as np  
import torch, os
import torch.utils.data as data

from tqdm import tqdm
from os.path import join, basename, dirname

from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, 
                                        adjust_hue, adjust_saturation, normalize)

class Datagen_teeth(data.Dataset):
    def __init__(self, opt):
        super(Datagen_teeth, self).__init__()
        self.opt = opt
        
        self.mean = opt['mean']
        self.std = opt['std']
        self.resize = opt['resize']
        
        
        self.filelist = self.get_filelists()
        print(f'info: dataset size: {len(self.filelist)}')
        
    def get_filelists(self):
        """ filelist include file path.
            example: /user/vfhq/group/file.
        """
        
        filelist = []
        dataset_config = self.opt['dataset']
        
        if 'vfhq' in dataset_config:
            data_root = dataset_config['vfhq'].get('dpath')
            filelist_root = dataset_config['vfhq'].get('filelist')
            latent_code_root = dataset_config['vfhq'].get('latent_code')
            
            vfhq_latent_code = np.load(latent_code_root)
            with open(filelist_root, 'r') as f:
                for line in tqdm(f, desc='read vfhq filelist.'):
                    line = line.strip()
                    filelist.append(join(data_root, line))
                    
        return filelist, vfhq_latent_code
    
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
    
    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, idx):
        
        
        
        return data_dict
