import cv2
import numpy as np
import os
from tqdm import tqdm
from skimage.util import img_as_ubyte
import torch.nn as nn
import torch
import torch.nn.functional as F
from natsort import natsorted
from glob import glob
from adamp import AdamP
from restormer import Restormer
def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

####### Load yaml #######

model_restoration = Restormer().cuda()
model_restoration = nn.DataParallel(model_restoration, device_ids=[0])#########
optimizer = AdamP(model_restoration.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-4)
model_root = 'model/ckpt/v22.pth'
checkpoint = torch.load(model_root)
model_restoration.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_dict'])
epoch = checkpoint['epoch']
print("===>Testing using weights: ", model_root)
model_restoration.eval()
result_dir = "result/v22/syn" #real or syn
if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
# if not os.path.exists(result_dir_rain):
#         os.makedirs(result_dir_rain, exist_ok=True)
factor = 8
datasets = ['val_real']
for dataset in datasets:

    # inp_dir = os.path.join("./data/real200") #####
    inp_dir = os.path.join("./data/Rain200H/test/input")  ####
    files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))
    with torch.no_grad():
        for file_ in tqdm(files):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            img = np.float32(load_img(file_))/255.
            img = torch.from_numpy(img).permute(2,0,1)
            input_ = img.unsqueeze(0).cuda()
            # Padding in case images are not multiples of 8
            h,w = input_.shape[2], input_.shape[3]
            H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_ = F.pad(input_, (0,padw,0,padh), 'reflect')
            restored, rain_layer = model_restoration(input_)
            # Unpad images to original dimensions
            restored = restored[:,:,:h,:w]
            # rain_layer = rain_layer[:,:,:h,:w]
            restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            # rain_layer = torch.clamp(rain_layer, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0]+'.png')), img_as_ubyte(restored))
            # save_img((os.path.join(result_dir_rain, os.path.splitext(os.path.split(file_)[-1])[0] + '.png')),img_as_ubyte(rain_layer))