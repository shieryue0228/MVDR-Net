import os
import torch
from glob import glob
from os.path import join
from torchvision.transforms import transforms

# initialize the reliable bank

input_dir = './data/real3000' 
result_dir = './data/candidate'
input_lists = glob(join(input_dir, '*.*'))
if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
for gen_path in zip(input_lists):
    img = torch.zeros((3,128,128))
    img_name = gen_path[0].split('/')[3]
    print(img_name)
    toPil = transforms.ToPILImage()
    res = toPil(img).convert('RGB')
    res.save(os.path.join(result_dir, img_name))
