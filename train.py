import os
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset_all import TrainUnlabeled, ValLabeled, DataLoaderTrain
from restormer import Restormer
from utils import *
from trainer_v22 import Trainer

def main(gpu, args):
    args.local_rank = gpu
    # random seed
    setup_seed(2022)
    # load data
    paired_dataset = DataLoaderTrain(dataroot='./data/Rain200H/', phase='train', finesize=args.crop_size)#
    unpaired_dataset = TrainUnlabeled(dataroot='./data/real1800/', phase='train', finesize=args.crop_size)   ########### 每次需要调整candidate目录
    val_dataset = ValLabeled(dataroot='./data/Rain200H/', phase='test', finesize=args.crop_size)
    paired_sampler = None
    unpaired_sampler = None
    val_sampler = None
    paired_loader = DataLoader(paired_dataset, batch_size=args.train_batchsize, sampler=paired_sampler, shuffle=True, num_workers=0, drop_last=False, pin_memory=True)
    unpaired_loader = DataLoader(unpaired_dataset, batch_size=args.train_batchsize, sampler=unpaired_sampler, shuffle=True, num_workers=0, drop_last=False, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batchsize, sampler=val_sampler, shuffle=True, num_workers=0, drop_last=False, pin_memory=True)
    print('there are total %s batches for train' % (len(paired_loader)))
    print('there are total %s batches for un_train' % (len(unpaired_loader)))
    print('there are total %s batches for val' % (len(val_loader)))
    # create model
    net = Restormer()
    ema_net = Restormer()
    ema_net = create_emamodel(ema_net)
    print('student model params: %d' % count_parameters(net))
    # tensorboard
    writer = SummaryWriter(log_dir=args.log_dir)
    trainer = Trainer(model=net, tmodel=ema_net, args=args, supervised_loader=paired_loader,
                      unsupervised_loader=unpaired_loader,
                      val_loader=val_loader, iter_per_epoch=len(unpaired_loader), writer=writer)
    trainer.train()
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-g', '--gpus', default=1, type=int, metavar='N')
    parser.add_argument('--num_epochs', default=400, type=int)  #Rain200H/L:400   DID:200
    parser.add_argument('--train_batchsize', default=2, type=int, help='train batchsize')
    parser.add_argument('--val_batchsize', default=2, type=int, help='val batchsize')
    parser.add_argument('--crop_size', default=128, type=int, help='crop size')
    parser.add_argument('--resume', default='False', type=str, help='if resume')
    parser.add_argument('--resume_path', default='', type=str, help='if resume')
    parser.add_argument('--use_pretain', default='False', type=str, help='use pretained model')
    parser.add_argument('--pretrained_path', default='/path/to/pretained/net.pth', type=str, help='if pretrained')
    parser.add_argument('--save_path', default='./model/ckpt/', type=str)
    parser.add_argument('--log_dir', default='./model/log/v22', type=str)   ##################
    parser.add_argument('--pseudo_save_dir', type=str, default='./pseudo_labels_vis_v22', help='Path to save pseudo labels') ##################

    args = parser.parse_args()
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    main(-1, args)
