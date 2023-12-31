from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
import logging
from torch.utils.data.dataloader import DataLoader
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
import cv2
import os
from utils import setup_logging

logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


class Trainer():
    def __init__(self, args, model, optimizer, scheduler, dataset_train, dataset_eval):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataset_train = dataset_train
        self.dataset_eval = dataset_eval
        self.cur_epoch = 0
        self.cur_global_step = 0

        self.logger = SummaryWriter(os.path.join(args.exp_root_dir, 'logs', args.exp_name))
        setup_logging(args.exp_name, args.exp_root_dir)

        self.build_dataloader()

        if args.load_state: # resume training
            self.load_state(args.load_state_exp_name, args.load_state_epoch)
        self.build_accelerator()
        if args.load_state: # accelerate.logger.info should be called after accelerator is built
            logger.info(f'Resume training from experiment {args.load_state_exp_name}, epoch {self.cur_epoch-1}, global step {self.cur_global_step}')


    def build_dataloader(self):
        self.dataloader_train = DataLoader(self.dataset_train, batch_size=self.args.batch_size, pin_memory=True, \
                                           shuffle=True, num_workers=self.args.num_workers)
        self.dataloader_eval = DataLoader(self.dataset_eval, batch_size=self.args.eval_batch_size, pin_memory=True, \
                                          shuffle=True,num_workers=self.args.num_workers)

    def build_accelerator(self):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            mixed_precision=self.args.mixed_precision, 
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=self.args.gradient_accumulate_steps
            )
        self.model, self.optimizer, self.scheduler, self.dataloader_train, self.dataloader_eval \
         = self.accelerator.prepare(self.model, self.optimizer, self.scheduler, self.dataloader_train, self.dataloader_eval)
        
    def save_state(self):
        '''
        Save the state of the model, optimizer and scheduler only in the main process.
        '''
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            optimizer_state = self.accelerator.unwrap_model(self.optimizer).state_dict()
            if self.scheduler is not None:
                scheduler_state =  self.accelerator.unwrap_model(self.scheduler).state_dict()
            else:
                scheduler_state = None
            
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            model_state = unwrapped_model.get_trainable_parameters()

            torch.save({
                'epoch': self.cur_epoch,
                'global_step': self.cur_global_step,
                'optimizer_state': optimizer_state,
                'scheduler_state': scheduler_state,
                'model_state': model_state,
            }, os.path.join(self.args.exp_root_dir, 'ckpts', self.args.exp_name, f'epoch_{self.cur_epoch}.pth'))

    def load_state(self, exp_name, epoch):
        '''
        Load state from a checkpoint before constructing the accelerator.
        '''
        path = os.path.join(self.args.exp_root_dir, 'ckpts', exp_name, f'epoch_{epoch}.pth')
        assert os.path.exists(path), f'Checkpoint {path} does not exist.'

        #TODO: Add DDP support
        
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.cur_epoch = checkpoint['epoch'] + 1
        self.cur_global_step = checkpoint['global_step']
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        # update partial states
        self.model.load_state_dict(checkpoint['model_state'], strict=False)

    def train_epoch(self, use_tqdm=True, inner_collect_fn=None):
        '''
        Single epoch training
        '''

        pbar = tqdm(total=len(self.dataloader_train), desc=f'Training in epoch {self.cur_epoch}', disable=not use_tqdm or not self.accelerator.is_main_process, \
                    iterable=enumerate(self.dataloader_train))
        
        for step, batch in pbar:
            with self.accelerator.accumulate(self.model):
                loss = self.model(batch)['loss_total']
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.accelerator.is_main_process:
                self.logger.add_scalar('loss', loss.item(), self.cur_global_step)

            self.cur_global_step += 1

    def train(self, use_tqdm=True, inner_collect_fn=None):
        '''
        Train the model and save the checkpoint per epoch
        '''
        logger.info('Start training...')
        pbar = tqdm(total=self.args.num_epochs, desc='Epoch', disable=not use_tqdm or not self.accelerator.is_main_process, iterable=range(self.args.num_epochs))
        pbar.update(self.cur_epoch)
        self.model.train()
        # self.eval()
        # exit(0)

        for epoch in pbar:
            self.train_epoch(use_tqdm=use_tqdm, inner_collect_fn=inner_collect_fn)

            if self.scheduler is not None:
                self.scheduler.step()

            if (self.cur_epoch+1) % self.args.save_state_epoch == 0:
                self.save_state()
            self.eval()

            self.cur_epoch += 1


    @torch.no_grad()
    def eval(self, use_tqdm=True, inner_collect_fn=None):
        logger.info(f'Start evaluation at global step {self.cur_global_step}, epoch {self.cur_epoch}...')
        self.model.eval()
        pbar = tqdm(total=self.args.eval_image_num, desc='Eval per epoch', disable=not use_tqdm, \
                    iterable=enumerate(self.dataloader_eval))

        for step, batch in pbar:
            if step >= self.args.eval_image_num:
                break
            res = self.model(batch)
            pred_image = res['logits_imgs']
            target_pose = batch['target_pose_image']
            source_image = batch['source_image']
            diffusion_target_image = batch['diffusion_target_image']

            # gather the results from all GPUs
            pred_image, target_pose, source_image, diffusion_target_image = \
            self.accelerator.gather_for_metrics(pred_image), self.accelerator.gather_for_metrics(target_pose), \
            self.accelerator.gather_for_metrics(source_image), self.accelerator.gather_for_metrics(diffusion_target_image)

            if self.accelerator.is_main_process: # only save the results in the main process
                pred_image = (pred_image*255).cpu().numpy().round().astype('uint8').transpose((0,2,3,1))
                target_pose = (target_pose*255).cpu().numpy().round().astype('uint8').transpose((0,2,3,1))
                source_image = source_image.cpu().numpy()
                diffusion_target_image = ((diffusion_target_image * 0.5 + 0.5)*255).cpu().numpy().round().astype('uint8').transpose((0,2,3,1))

                line = 255*np.ones((pred_image.shape[0],pred_image.shape[1],5,3))
                img = np.concatenate([source_image,line, target_pose,line,diffusion_target_image,line,pred_image], axis=2)

            
                os.makedirs(os.path.join(self.args.exp_root_dir, 'results', self.args.exp_name, f'epoch_{self.cur_epoch}_global_step_{self.cur_global_step}'), exist_ok=True)
                for i in range(self.args.eval_batch_size):
                    cv2.imwrite(os.path.join(self.args.exp_root_dir, 'results', self.args.exp_name, f'epoch_{self.cur_epoch}_global_step_{self.cur_global_step}',f'eval_{step * self.args.eval_batch_size  + i}.jpg'), img[i])
            

        self.model.train()
        

        
class WarmupLinearLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        max_iter,
        min_lr=1e-8,
        warmup_ratio=0.1,
        last_epoch=-1,
    ):
        self.max_iter = max_iter
        self.min_lr = min_lr
        self.warmup_ratio = warmup_ratio
        self.warmup_iters = int(warmup_ratio*max_iter)
        super(WarmupLinearLR, self).__init__(optimizer, last_epoch)

    def get_lr_factor(self):
        tot_step = self.max_iter
        warmup_step = self.warmup_iters
        step = self.last_epoch
        if step < warmup_step:
            return max(0, step / warmup_step)
        elif step > tot_step:
            step = tot_step
        return max(0, (tot_step-step)/(tot_step-warmup_step))

    def get_lr(self):
        warmup_factor = self.get_lr_factor()
        return [
            max(self.min_lr, base_lr * warmup_factor)
            for base_lr in self.base_lrs
        ]


if __name__ == '__main__':
    pass