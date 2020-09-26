import os
import torch
import torch.utils.data as data
from torch.autograd import Variable as V
from torchvision.transforms import transforms
import numpy as np
import skimage.io as io
import random

def img_loader(root, path):
    cloud_img = io.imread(f'{os.path.join(root, path[0], path[2:])}_cloud.tif') / 10000.
    mask_img = io.imread(f'{os.path.join(root, path[0], path[2:])}_mask.tif')[:,:, np.newaxis] / 255.
    sar_img = io.imread(f'{os.path.join(root, path[0], path[2:])}_sar.tif')
    clean_img = io.imread(f'{os.path.join(root, path[0], path[2:])}_clean.tif') / 10000.
    M = np.clip((clean_img-cloud_img).sum(axis=2), 0, 1).astype(np.float32)
    
    if random.random()<0.5:
        cloud_img = np.flipud(cloud_img)
        mask_img = np.flipud(mask_img)
        sar_img = np.flipud(sar_img)
        clean_img = np.flipud(clean_img)
        M = np.flipud(M)
    
    if random.random()<0.5:
        cloud_img = np.fliplr(cloud_img)
        mask_img = np.fliplr(mask_img)
        sar_img = np.fliplr(sar_img)
        clean_img = np.fliplr(clean_img)
        M = np.fliplr(M)

    if random.random()<0.5:
        cloud_img = np.rot90(cloud_img)
        mask_img = np.rot90(mask_img)
        sar_img = np.rot90(sar_img)
        clean_img = np.rot90(clean_img)
        M = np.rot90(M)

    # if random.random()<0.25:
    #     scale = random.uniform(1,1.5)
    #     pct = random.uniform(0,1)
    #     img = img.zoom(scale, row_pct=pct, col_pct=pct)
    #     mask = mask.zoom(scale, row_pct=pct, col_pct=pct)

    return cloud_img, sar_img, clean_img, mask_img, M

class SemanticFolder(data.Dataset):
    def __init__(self, root, paths):
        self.ids = np.arange(len(paths)) 
        self.root = root
        self.paths = paths
        self.loader = img_loader
        # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
        # self.normalize = transforms.Normalize(mean=[0.3858, 0.4537, 0.3683], std=[0.1603, 0.1247, 0.1279]) # guizhou
        # self.normalize = norm_func
        # self.normalize = transforms.Normalize(mean=[0.3632, 0.3494, 0.2943], std=[0.4815, 0.4771, 0.4564])
    def __getitem__(self, index):
        id = self.ids[index]
        cloud_img, sar_img, clean_img, mask_img, M = self.loader(self.root, self.paths[id])
        # img = img.data * 255
        # img /= 255.
        cloud_img = torch.from_numpy(cloud_img.transpose(2,0,1).astype('float32'))
        sar_img = torch.from_numpy(sar_img.transpose(2,0,1).astype('float32'))
        # full_img = torch.cat((cloud_img, sar_img), dim=0)
        clean_img = torch.from_numpy(clean_img.transpose(2,0,1).astype('float32'))
        mask_img = torch.from_numpy(mask_img.transpose(2,0,1).astype('float32'))
        M = torch.from_numpy(M.astype('float32'))
        # img = self.normalize(img)
        #img -= self.mean
        #img /= self.std
        return cloud_img, clean_img, M
        # return cloud_img, clean_img, M
    def __len__(self):
        return self.ids.shape[0]