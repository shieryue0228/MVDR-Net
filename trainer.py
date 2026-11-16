from tqdm import tqdm
from torch.optim import lr_scheduler
import PIL.Image as Image
from loss import losses
from utils import *
from torch.autograd import Variable
from loss.losses import *
import pyiqa
from itertools import cycle
class Trainer:
    def __init__(self, model, tmodel, args, supervised_loader, unsupervised_loader, val_loader, iter_per_epoch, writer):

        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.val_loader = val_loader
        self.args = args
        self.iter_per_epoch = iter_per_epoch
        self.writer = writer
        self.model = model
        self.tmodel = tmodel
        self.gamma = 0.5
        self.start_epoch = 1
        self.epochs = args.num_epochs
        self.save_period = 1
        self.loss_unsup = TVLoss()
        self.loss_L1 = nn.L1Loss()
        #Restormer
        self.loss_fft = losses.fftLoss()
        self.loss_Edge = losses.EdgeLoss()

        self.loss_rain = nn.L1Loss()
        self.loss_lcr = LocalContrastLoss()
        self.consistency = 0.2
        self.consistency_rampup = 200.0
        self.iqa_metric = pyiqa.create_metric('arniqa', device='cuda', as_loss=False)

        self.curiter = 0
        self.model.cuda()
        self.tmodel.cuda()
        self.device, available_gpus = self._get_available_devices(self.args.gpus)
        self.model = torch.nn.DataParallel(self.model, device_ids=available_gpus)

        # set optimizer and learning rate
        self.optimizer_s = torch.optim.AdamW(self.model.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-4)
        self.lr_scheduler_s = lr_scheduler.CosineAnnealingLR(self.optimizer_s, eta_min=2e-6, T_max=267)
        self.pseudo_save_dir = args.pseudo_save_dir
        os.makedirs(self.pseudo_save_dir, exist_ok=True)

    @torch.no_grad()
    def update_teachers(self, teacher, itera, keep_rate=0.996):
        # exponential moving average(EMA)
        alpha = min(1 - 1 / (itera + 1), keep_rate)
        for ema_param, param in zip(teacher.parameters(), self.model.parameters()):
            ema_param.data = (alpha * ema_param.data) + (1 - alpha) * param.data

    def predict_with_out_grad(self, image):
        with torch.no_grad():
            predict_target_ul = self.tmodel(image)

        return predict_target_ul

    def freeze_teachers_parameters(self):
        for p in self.tmodel.parameters():
            p.requires_grad = False

    def get_reliable(self, fusion, student_predict, positive_list, p_name):
        N = fusion.shape[0]#[2, 3, 128, 128]
        s_p = student_predict.clamp(0, 1)
        score_t = self.iqa_metric(fusion)
        score_s = self.iqa_metric(s_p)
        score_r = self.iqa_metric(positive_list)
        positive_sample = positive_list.clone()
        for idx in range(0, N):
            if score_t[idx] > score_s[idx]:
                if score_t[idx] > score_r[idx]:
                    positive_sample[idx] = fusion[idx]
                    # update the reliable bank
                    temp_c = np.transpose(fusion[idx].detach().cpu().numpy(), (1, 2, 0))
                    temp_c = np.clip(temp_c, 0, 1)
                    arr_c = (temp_c * 255).astype(np.uint8)
                    arr_c = Image.fromarray(arr_c)
                    arr_c.save('%s' % p_name[idx])
        del N, score_r, score_s, score_t, fusion, student_predict, positive_list
        return positive_sample

    def train(self):
        self.freeze_teachers_parameters()#

        if self.start_epoch == 1:
            initialize_weights(self.model)
        else:
            checkpoint = torch.load(self.args.resume_path)
            self.model.load_state_dict(checkpoint['state_dict'])
        best_val_psnr = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            loss_ave, psnr_train = self._train_epoch(epoch)
            loss_val = loss_ave.item() / self.args.crop_size * self.args.train_batchsize
            train_psnr = sum(psnr_train) / len(psnr_train)
            psnr_val = self._valid_epoch(max(0, epoch))
            val_psnr = sum(psnr_val) / len(psnr_val)

            print('[%d] main_loss: %.6f, train psnr: %.6f, val psnr: %.6f, lr: %.8f' % (
                epoch, loss_val, train_psnr, val_psnr, self.lr_scheduler_s.get_last_lr()[0]))

            # Save checkpoint
            if epoch % self.save_period == 0 and self.args.local_rank <= 0 and val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                state = {'arch': type(self.model).__name__,
                         'epoch': epoch,
                         'state_dict': self.model.state_dict(),
                         'optimizer_dict': self.optimizer_s.state_dict()}
                ckpt_name = str(self.args.save_path) + 'model.pth'
                print("Saving a checkpoint: {} ...".format(str(ckpt_name)))
                torch.save(state, ckpt_name)

    def _train_epoch(self, epoch):
        sup_loss = AverageMeter()
        unsup_loss = AverageMeter()
        loss_total = AverageMeter()
        loss_total_ave = 0.0
        psnr_train = []
        self.model.train()
        self.freeze_teachers_parameters()

        train_loader = iter(zip(self.supervised_loader, cycle(self.unsupervised_loader)))
        tbar = range(len(self.supervised_loader))
        tbar = tqdm(tbar, ncols=130, leave=True)
        for i in tbar:
            (img_data, label, rain_layer, rain_layer_ul), (unpaired_data_w, unpaired_data_s1, unpaired_data_s2, p_list, p_name) = next(train_loader)

            img_data = Variable(img_data).cuda(non_blocking=True)
            label = Variable(label).cuda(non_blocking=True)
            rain_layer_ul = Variable(rain_layer_ul).cuda(non_blocking=True)
            unpaired_data_s1 = Variable(unpaired_data_s1).cuda(non_blocking=True)
            unpaired_data_s2 = Variable(unpaired_data_s2).cuda(non_blocking=True)
            unpaired_data_w = Variable(unpaired_data_w).cuda(non_blocking=True)
            p_list = Variable(p_list).cuda(non_blocking=True) #candidate

            assert rain_layer_ul.size(0) >= 2, "rain_layer batch size must >= 2"
            rain1 = rain_layer_ul[0:1].expand_as(unpaired_data_s1)  # [1,3,H,W]
            rain2 = rain_layer_ul[1:2].expand_as(unpaired_data_s2)  # [1,3,H,W]

            # rainy = background + alpha * rain_layer
            unpaired_data_s1 = (unpaired_data_s1 + 0.2 * rain1).clamp(0, 1)
            unpaired_data_s2 = (unpaired_data_s2 + 0.2 * rain2).clamp(0, 1)

            # teacher output
            outputs_ul1, _ = self.predict_with_out_grad(unpaired_data_s1)
            outputs_ul2, _ = self.predict_with_out_grad(unpaired_data_s2)
            outputs_ul3, _ = self.predict_with_out_grad(unpaired_data_w)

            # student output
            outputs_l, outputs_l_rain = self.model(img_data) #有标签的学生网络输出
            outputs_ul, outputs_ul_rain = self.model(unpaired_data_w)
            # label loss
            L1_loss = self.loss_L1(outputs_l, label)

            fft_loss = self.loss_fft(outputs_l, label)
            edge_loss = self.loss_Edge(outputs_l, label)
            loss_sup = L1_loss + 0.01 * fft_loss + edge_loss
            sup_loss.update(loss_sup.mean().item())

            fusion = self.mean_fusion_3frames(outputs_ul1, outputs_ul2, outputs_ul3).clamp(0, 1)
            p_sample = self.get_reliable(fusion, outputs_ul, p_list, p_name)

            if epoch % 10 == 0 and i == 0:
                self.save_pseudo_labels(
                    inputs=[unpaired_data_w, unpaired_data_s1, unpaired_data_s2],
                    preds=[outputs_ul1, outputs_ul2, outputs_ul3],
                    p_sample=p_sample,
                    epoch=epoch
                )

            loss_unsu = self.loss_unsup(outputs_ul) + 0.1 * self.loss_lcr(outputs_ul, p_sample, unpaired_data_w)
            unsup_loss.update(loss_unsu.mean().item())
            consistency_weight = self.get_current_consistency_weight(epoch)
            total_loss = consistency_weight * loss_unsu + loss_sup
            total_loss = total_loss.mean()
            loss_total.update(total_loss.item())

            psnr_train.extend(to_psnr(outputs_l, label))

            self.optimizer_s.zero_grad()
            total_loss.backward()
            self.optimizer_s.step()

            tbar.set_description('Train-Student Epoch {} | Ls {:.4f} Lu {:.4f}|'
                                 .format(epoch, sup_loss.avg, unsup_loss.avg))

            del img_data, label, unpaired_data_w, unpaired_data_s1, unpaired_data_s2 #用来尽早释放 Python 变量引用，帮助显存回收
            with torch.no_grad():
                self.update_teachers(teacher=self.tmodel, itera=self.curiter)
                self.curiter = self.curiter + 1

        loss_total_ave = loss_total_ave + total_loss

        self.writer.add_scalar('consistency_weight', self.get_current_consistency_weight(epoch), global_step=epoch)
        self.writer.add_scalar('Train_loss', loss_total.avg, global_step=epoch)
        self.writer.add_scalar('sup_loss', sup_loss.avg, global_step=epoch)
        self.writer.add_scalar('unsup_loss', unsup_loss.avg, global_step=epoch)
        if epoch-1 > 133:
            self.lr_scheduler_s.step()
        return loss_total_ave, psnr_train

    def _valid_epoch(self, epoch):
        psnr_val = []
        self.model.eval()
        self.tmodel.eval()
        val_psnr = AverageMeter()
        val_ssim = AverageMeter()
        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            for i, (val_data, val_label) in enumerate(tbar):
                val_data = Variable(val_data).cuda()
                val_label = Variable(val_label).cuda()

                # forward
                val_output, _ = self.model(val_data)
                temp_psnr, temp_ssim, N = compute_psnr_ssim(val_output, val_label)
                val_psnr.update(temp_psnr, N)
                val_ssim.update(temp_ssim, N)
                psnr_val.extend(to_psnr(val_output, val_label))
                tbar.set_description('{} Epoch {} | PSNR: {:.4f}, SSIM: {:.4f}|'.format(
                    "Eval-Student", epoch, val_psnr.avg, val_ssim.avg))

            self.writer.add_scalar('Val_psnr', val_psnr.avg, global_step=epoch)
            self.writer.add_scalar('Val_ssim', val_ssim.avg, global_step=epoch)
            del val_output, val_label, val_data
            return psnr_val

    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            print('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            print(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        available_gpus = list(range(n_gpu))
        return device, available_gpus

    def get_current_consistency_weight(self, epoch):
        return self.consistency * self.sigmoid_rampup(epoch, self.consistency_rampup)

    def sigmoid_rampup(self, current, rampup_length):
        # Exponential rampup
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    def mean_fusion_3frames(self, img1: torch.Tensor, img2: torch.Tensor, img3: torch.Tensor) -> torch.Tensor:

        stacked = torch.stack([img1, img2, img3], dim=1)
        p_sample = torch.mean(stacked, dim=1)
        return p_sample
    @torch.no_grad()
    def save_pseudo_labels(self, inputs, preds, p_sample, epoch):

        from torchvision.utils import save_image
        save_dir = os.path.join(self.pseudo_save_dir, f'epoch_{epoch:03d}')
        os.makedirs(save_dir, exist_ok=True)
        idx = 0
        def denorm(tensor):
            return tensor.clamp(0, 1)
        for i, inp in enumerate(inputs):
            img = denorm(inp[idx:idx + 1])
            save_image(img, os.path.join(save_dir, f'input_aug{i + 1}.png'))
        for i, pred in enumerate(preds):
            img = denorm(pred[idx:idx + 1])
            save_image(img, os.path.join(save_dir, f'teacher_pred_aug{i + 1}.png'))
        p_img = denorm(p_sample[idx:idx + 1])
        save_image(p_img, os.path.join(save_dir, f'PSEUDO_LABEL.png'))

