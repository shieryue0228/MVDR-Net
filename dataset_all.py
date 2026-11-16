import os.path
import torchvision.transforms.functional as TF
import torch.utils.data as data
from PIL import Image
import numpy as np
import random
from random import randrange
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torch
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
from scipy.ndimage import gaussian_filter

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def rotate(img,rotate_index):
    '''
    :return: 8 version of rotating image
    '''
    if rotate_index == 0:
        return img
    if rotate_index==1:
        return img.rotate(90)
    if rotate_index==2:
        return img.rotate(180)
    if rotate_index==3:
        return img.rotate(270)
    if rotate_index==4:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    if rotate_index==5:
        return img.rotate(90).transpose(Image.FLIP_TOP_BOTTOM)
    if rotate_index==6:
        return img.rotate(180).transpose(Image.FLIP_TOP_BOTTOM)
    if rotate_index==7:
        return img.rotate(270).transpose(Image.FLIP_TOP_BOTTOM)


class DataLoaderTrain(data.Dataset):
    def __init__(self, dataroot, phase, finesize):
        super(DataLoaderTrain, self).__init__()
        self.phase = phase
        self.root = dataroot
        self.ps = finesize

        self.dir_A = os.path.join(self.root, self.phase , 'input')#./data/target\labeled/input   dataroot='D://Raindataset/Rain200H/', phase='train'
        self.dir_B = os.path.join(self.root, self.phase , 'target')
        self.dir_C = os.path.join(self.root, self.phase , 'rain_layer')
        print(self.dir_A)
        # image path
        self.A_paths = sorted(make_dataset(self.dir_A))#./data/target\\labeled/input\\1.png
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.C_paths = sorted(make_dataset(self.dir_C))

        self.inp_filenames = [os.path.join(x) for x in self.A_paths if is_image_file(x)]
        self.tar_filenames = [os.path.join(x) for x in self.B_paths if is_image_file(x)]
        self.lay_filenames = [os.path.join(x) for x in self.C_paths if is_image_file(x)]
        self.sizex = len(self.tar_filenames)  # get the size of target

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]
        lay_path = self.lay_filenames[index_]

        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)
        lay_img = Image.open(lay_path)

        w, h = tar_img.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)
        lay_img = TF.to_tensor(lay_img)
        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')
            lay_img = TF.pad(lay_img, (0, 0, padw, padh), padding_mode='reflect')

        aug = random.randint(0, 2)
        if aug == 1:
            inp_img = TF.adjust_gamma(inp_img, 1)
            tar_img = TF.adjust_gamma(tar_img, 1)
            lay_img = TF.adjust_gamma(lay_img, 1)

        aug = random.randint(0, 2)
        if aug == 1:
            sat_factor = 1 + (0.2 - 0.4 * np.random.rand())
            inp_img = TF.adjust_saturation(inp_img, sat_factor)
            tar_img = TF.adjust_saturation(tar_img, sat_factor)
            lay_img = TF.adjust_saturation(lay_img, sat_factor)



        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)
        aug = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]
        lay_img = lay_img[:, rr:rr + ps, cc:cc + ps]

        # Data Augmentations
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
            lay_img = lay_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
            lay_img = lay_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
            lay_img = torch.rot90(lay_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
            lay_img = torch.rot90(lay_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
            lay_img = torch.rot90(lay_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
            lay_img = torch.rot90(lay_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))
            lay_img = torch.rot90(lay_img.flip(2), dims=(1, 2))

        return inp_img, tar_img, lay_img, lay_img



# class TrainUnlabeled(data.Dataset):
#     def __init__(self, dataroot, phase, finesize):
#         super().__init__()
#         self.phase = phase
#         self.root = dataroot
#         self.fineSize = finesize
#
#         self.dir_A = os.path.join(self.root, self.phase + '/input')
#         self.dir_D = os.path.join(self.root, self.phase + '/candidate')
#
#         # image path
#         self.A_paths = sorted(make_dataset(self.dir_A))
#         self.D_paths = sorted(make_dataset(self.dir_D))
#
#         # transform
#         self.transform = ToTensor()  # [0,1]
#
#     def __getitem__(self, index):
#         A = Image.open(self.A_paths[index]).convert("RGB")
#         candidate = Image.open(self.D_paths[index]).convert('RGB')
#         #原始处理
#         A = TF.center_crop(A, (128, 128))
#         A = A.resize((self.fineSize, self.fineSize), Image.ANTIALIAS)
#
#         # strong augmentation
#         # strong_data = data_aug(A)  #数据增强
#         strong_data = A
#         tensor_w = self.transform(A) #weakly
#         tensor_s = self.transform(strong_data) #strong
#         tensor_d = self.transform(candidate)
#         name = self.D_paths[index]
#
#         return tensor_w, tensor_s, tensor_d, name
#
#     def __len__(self):
#         return len(self.A_paths)
class TrainUnlabeled(data.Dataset):
    def __init__(self, dataroot, phase, finesize):
        super().__init__()
        self.phase = phase
        self.root = dataroot
        self.fineSize = finesize

        # self.dir_A = os.path.join(self.root, self.phase + '/input')
        # self.dir_D = os.path.join(self.root, self.phase + '/candidate')

        self.dir_A = os.path.join(self.root)  # dataroot='./data/real1800/'
        self.dir_D = os.path.join('./data/candidate_v22') ##################

        # image path
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.D_paths = sorted(make_dataset(self.dir_D))
        # transform
        self.transform = ToTensor()  # [0,1]

    def __getitem__(self, index):
        A = Image.open(self.A_paths[index]).convert("RGB")
        candidate = Image.open(self.D_paths[index]).convert('RGB')

        # strong augmentation
        inp_img = TF.to_tensor(A)
        ps = self.fineSize
        w, h = A.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0
        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')

        # Crop patch
        A = TF.center_crop(inp_img, (self.fineSize, self.fineSize))
        tensor_w = A
        tensor_s1 = A
        tensor_s2 = A
        tensor_d = self.transform(candidate)
        name = self.D_paths[index]

        return tensor_w, tensor_s1, tensor_s2, tensor_d, name

    def __len__(self):
        return len(self.A_paths)

class ValLabeled(data.Dataset):
    def __init__(self, dataroot, phase, finesize):
        super().__init__()
        self.phase = phase
        self.root = dataroot
        self.fineSize = finesize

        self.dir_A = os.path.join(self.root, self.phase , 'input')
        self.dir_B = os.path.join(self.root, self.phase , 'target')

        # image path
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))

        # transform
        self.transform = ToTensor()  # [0,1]

    def __getitem__(self, index):
        # A, B is the image pair, hazy, gt respectively
        A = Image.open(self.A_paths[index]).convert("RGB")
        B = Image.open(self.B_paths[index]).convert("RGB")
        # resized_a = A.resize((self.fineSize, self.fineSize), Image.Resampling.LANCZOS)
        # resized_b = B.resize((self.fineSize, self.fineSize), Image.Resampling.LANCZOS)

        resized_a = TF.center_crop(A, (self.fineSize, self.fineSize))
        resized_b = TF.center_crop(B, (self.fineSize, self.fineSize))

        # transform to (0, 1)
        tensor_a = self.transform(resized_a)
        tensor_b = self.transform(resized_b)

        return tensor_a, tensor_b

    def __len__(self):
        return len(self.A_paths)

class DataLoaderTrain_v9(data.Dataset):
    def __init__(self, dataroot, phase, finesize):
        super(DataLoaderTrain_v9, self).__init__()
        self.phase = phase
        self.root = dataroot
        self.ps = finesize

        self.dir_A = os.path.join(self.root, self.phase , 'input')#./data/target\labeled/input   dataroot='D://Raindataset/Rain200H/', phase='train'
        self.dir_B = os.path.join(self.root, self.phase , 'target')
        self.dir_C = os.path.join(self.root, self.phase , 'rain_layer')

        # image path
        self.A_paths = sorted(make_dataset(self.dir_A))#./data/target\\labeled/input\\1.png
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.C_paths = sorted(make_dataset(self.dir_C))

        self.inp_filenames = [os.path.join(x) for x in self.A_paths if is_image_file(x)]
        self.tar_filenames = [os.path.join(x) for x in self.B_paths if is_image_file(x)]
        self.lay_filenames = [os.path.join(x) for x in self.C_paths if is_image_file(x)]
        self.sizex = len(self.tar_filenames)  # get the size of target

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]
        lay_path = self.lay_filenames[index_]

        # ===== 新增：随机选择一个不同的雨层 =====
        rand_index = random.randint(0, self.sizex - 1)
        while rand_index == index_:
            rand_index = random.randint(0, self.sizex - 1)
        lay_path2 = self.lay_filenames[rand_index]
        # =====================================

        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)
        lay_img = Image.open(lay_path)
        lay_img2 = Image.open(lay_path2)

        w, h = tar_img.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)
        lay_img = TF.to_tensor(lay_img)
        lay_img2 = TF.to_tensor(lay_img2)
        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')
            lay_img = TF.pad(lay_img, (0, 0, padw, padh), padding_mode='reflect')
            lay_img2 = TF.pad(lay_img2, (0, 0, padw, padh), padding_mode='reflect')

        aug = random.randint(0, 2)
        if aug == 1:
            inp_img = TF.adjust_gamma(inp_img, 1)
            tar_img = TF.adjust_gamma(tar_img, 1)
            lay_img = TF.adjust_gamma(lay_img, 1)
            lay_img2 = TF.adjust_gamma(lay_img2, 1)

        aug = random.randint(0, 2)
        if aug == 1:
            sat_factor = 1 + (0.2 - 0.4 * np.random.rand())
            inp_img = TF.adjust_saturation(inp_img, sat_factor)
            tar_img = TF.adjust_saturation(tar_img, sat_factor)
            lay_img = TF.adjust_saturation(lay_img, sat_factor)
            lay_img2 = TF.adjust_saturation(lay_img2, sat_factor)

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)
        aug = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]
        lay_img = lay_img[:, rr:rr + ps, cc:cc + ps]

        hh2, ww2 = lay_img2.shape[1], lay_img2.shape[2]
        rr2 = random.randint(0, hh2 - ps)
        cc2 = random.randint(0, ww2 - ps)
        lay_img2 = lay_img2[:, rr2:rr2 + ps, cc2:cc2 + ps]


        # Data Augmentations
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
            lay_img = lay_img.flip(1)
            lay_img2 = lay_img2.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
            lay_img = lay_img.flip(2)
            lay_img2 = lay_img2.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
            lay_img = torch.rot90(lay_img, dims=(1, 2))
            lay_img2 = torch.rot90(lay_img2, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
            lay_img = torch.rot90(lay_img, dims=(1, 2), k=2)
            lay_img2 = torch.rot90(lay_img2, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
            lay_img = torch.rot90(lay_img, dims=(1, 2), k=3)
            lay_img2 = torch.rot90(lay_img2, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
            lay_img = torch.rot90(lay_img.flip(1), dims=(1, 2))
            lay_img2 = torch.rot90(lay_img2.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))
            lay_img = torch.rot90(lay_img.flip(2), dims=(1, 2))
            lay_img2 = torch.rot90(lay_img2.flip(2), dims=(1, 2))

        return inp_img, tar_img, lay_img, lay_img, lay_img2

class TrainUnlabeled_v9(data.Dataset):
    def __init__(self, dataroot, phase, finesize):
        super().__init__()
        self.phase = phase
        self.root = dataroot
        self.fineSize = finesize

        self.dir_A = os.path.join(self.root)  # dataroot='./data/real1800/'
        self.dir_D = os.path.join('./data/candidate_v10') ##################

        # image path
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.D_paths = sorted(make_dataset(self.dir_D))
        # transform
        self.transform = ToTensor()  # [0,1]

    def __getitem__(self, index):
        A = Image.open(self.A_paths[index]).convert("RGB")
        candidate = Image.open(self.D_paths[index]).convert('RGB')

        # strong augmentation
        inp_img = TF.to_tensor(A)
        ps = self.fineSize
        w, h = A.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0
        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')

        # Crop patch
        A = TF.center_crop(inp_img, (self.fineSize, self.fineSize))
        tensor_w = A
        tensor_s1 = A
        tensor_s2 = A
        tensor_s3 = A
        tensor_s4 = A
        tensor_d = self.transform(candidate)
        name = self.D_paths[index]

        return tensor_w, tensor_s1, tensor_s2, tensor_s3, tensor_s4, tensor_d, name

    def __len__(self):
        return len(self.A_paths)

class TrainUnlabeled_v13(data.Dataset):
    def __init__(self, dataroot, phase, finesize):
        super().__init__()
        self.phase = phase
        self.root = dataroot
        self.fineSize = finesize
        self.dir_A = os.path.join(self.root)  # dataroot='./data/real1800/'
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.transform = ToTensor()  # [0,1]

    def __getitem__(self, index):
        A = Image.open(self.A_paths[index]).convert("RGB")

        # strong augmentation
        inp_img = TF.to_tensor(A)
        ps = self.fineSize
        w, h = A.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0
        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')

        # Crop patch
        A = TF.center_crop(inp_img, (self.fineSize, self.fineSize))
        tensor_w = A
        tensor_s1 = A
        tensor_s2 = A
        name = self.A_paths[index]

        return tensor_w, tensor_s1, tensor_s2, name

    def __len__(self):
        return len(self.A_paths)