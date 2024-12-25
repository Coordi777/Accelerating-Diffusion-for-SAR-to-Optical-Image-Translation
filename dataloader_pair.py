import argparse
import os

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

class pair_Dataset_csv_test(Dataset):
    def __init__(self, path_sar, path_opt, transforms_sar, transforms_opt):
        sar_imgs = pd.read_csv(path_sar)
        opt_imgs = pd.read_csv(path_opt)
        self.trans_sar = transforms_sar
        self.trans_opt = transforms_opt

        sar_path = sar_imgs.values.tolist()
        opt_path = opt_imgs.values.tolist()
        # 去掉多余维度
        for i in range(len(sar_path)):
            sar_path[i] = sar_path[i][0]
            opt_path[i] = opt_path[i][0]

        self.all_files_sar = sar_path
        self.all_files_opt = opt_path
        assert len(sar_path) == len(opt_path)
        print(f"loaded {len(sar_path)} datas")

    def __getitem__(self, index):
        img_path_sar = self.all_files_sar[index]
        img_path_opt = self.all_files_opt[index]

        return img_path_sar, img_path_opt

    def __len__(self):
        return len(self.all_files_sar)

class pair_Dataset_csv(Dataset):
    def __init__(self, path_sar, path_opt, transforms_sar, transforms_opt):
        sar_imgs = pd.read_csv(path_sar)
        opt_imgs = pd.read_csv(path_opt)
        self.trans_sar = transforms_sar
        self.trans_opt = transforms_opt

        sar_path = sar_imgs.values.tolist()
        opt_path = opt_imgs.values.tolist()
        # 去掉多余维度
        for i in range(len(sar_path)):
            sar_path[i] = sar_path[i][0]
            opt_path[i] = opt_path[i][0]

        self.all_files_sar = sar_path
        self.all_files_opt = opt_path
        assert len(sar_path) == len(opt_path)
        print(f"loaded {len(sar_path)} datas")

    def __getitem__(self, index):
        img_path_sar = self.all_files_sar[index]
        img_path_opt = self.all_files_opt[index]
        pil_img_sar = Image.open(img_path_sar).convert('L')
        pil_img_opt = Image.open(img_path_opt).convert('RGB')
        img_sar = self.trans_sar(pil_img_sar)
        img_opt = self.trans_opt(pil_img_opt)
        return img_sar, img_opt

    def __len__(self):
        return len(self.all_files_sar)


class pair_Dataset(Dataset):
    def __init__(self, path_sar, path_opt, transforms_sar, transforms_opt):
        sar_image_names = os.listdir(path_sar)
        sar_image_names.sort()
        opt_image_names = os.listdir(path_opt)
        opt_image_names.sort()

        assert len(sar_image_names) == len(opt_image_names)
        self.sar_list = []
        self.opt_list = []

        for img_name in sar_image_names:
            self.sar_list.append(os.path.join(path_sar, img_name))
            self.opt_list.append(os.path.join(path_opt, img_name))

        self.transform_sar = transforms_sar
        self.transform_opt = transforms_opt

    def __getitem__(self, index):
        img_path_sar = self.sar_list[index]
        img_path_opt = self.opt_list[index]
        pil_img_sar = Image.open(img_path_sar).convert('L')
        pil_img_opt = Image.open(img_path_opt).convert('RGB')

        img_sar = self.transform_sar(pil_img_sar)
        img_opt = self.transform_opt(pil_img_opt)
        return img_sar, img_opt

    def __len__(self):
        return len(self.opt_list)


class opt_Dataset(Dataset):
    def __init__(self, path_opt, transforms_opt):
        opt_image_names = os.listdir(path_opt)
        opt_image_names.sort()

        self.opt_list = []

        for img_name in opt_image_names:
            self.opt_list.append(os.path.join(path_opt, img_name))

        self.transform_opt = transforms_opt

    def __getitem__(self, index):
        img_path_opt = self.opt_list[index]
        pil_img_opt = Image.open(img_path_opt).convert('RGB')

        img_opt = self.transform_opt(pil_img_opt)
        return img_opt

    def __len__(self):
        return len(self.opt_list)

class opt_Dataset_csv(Dataset):
    def __init__(self, path_opt, transforms_opt):

        opt_imgs = pd.read_csv(path_opt)
        opt_path = opt_imgs.values.tolist()
        self.opt_list = []
        for i in range(len(opt_path)):
            opt_path[i] = opt_path[i][0]
        
        self.opt_list = opt_path
        print(f"Loaded {len(self.opt_list)} optical images.")


        self.transform_opt = transforms_opt

    def __getitem__(self, index):
        img_path_opt = self.opt_list[index]
        pil_img_opt = Image.open(img_path_opt).convert('RGB')

        img_opt = self.transform_opt(pil_img_opt)
        return img_opt

    def __len__(self):
        return len(self.opt_list)