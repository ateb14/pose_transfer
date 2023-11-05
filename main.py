from trainer import Trainer
from diffusion import Net
from dataset import DeepFashionDataset
import argparse, os
from yaml import safe_load
from accelerate.utils import set_seed
import torch


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/test.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        yaml_args = safe_load(f)
    parser.set_defaults(**yaml_args)
    args = parser.parse_args()

    train_dataset = DeepFashionDataset(args, 'train', args.data_root_dir)
    test_dataset = DeepFashionDataset(args, 'test', args.data_root_dir)

    model = Net(args)
    model.freeze_pretrained_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), weight_decay=args.decay)

    trainer = Trainer(args, model, optimizer, scheduler=None, dataset_train=train_dataset, dataset_eval=test_dataset)
    trainer.train()

