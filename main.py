from pathlib import Path
import argparse
import yaml
import torch
import os
import random
import numpy as np
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

from train import BasicTrain
from model.GraphTransformer import GraphTransformer
from datetime import datetime
from dataloader import init_dataloader


def main(args):
    with open(args.config_filename) as f:
        config = yaml.load(f, Loader=yaml.Loader)

        for i in range(args.repeat_time):
            dataloaders, node_size, node_feature_size, timeseries_size = \
                init_dataloader(config['data'])

            config['train']["seq_len"] = timeseries_size
            config['train']["node_size"] = node_size

            if 'GraphTransformer' == config['model']['type'] :
                model = GraphTransformer(config['model'], node_size,
                                         node_feature_size, timeseries_size, config['train']['pool_ratio'])
                use_train = BasicTrain


            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config['train']['lr'],
                weight_decay=config['train']['weight_decay'])

            opts = (optimizer,)

            loss_name = 'loss'
            if config['train']["group_loss"]:
                loss_name = f"{loss_name}_group_loss"
            if config['train']["sparsity_loss"]:
                loss_name = f"{loss_name}_sparsity_loss"


            save_folder_name = Path(config['train']['log_folder']) / Path(config['model']['type']) / Path(
                # date_time +
                f"{config['data']['dataset']}_{config['data']['atlas']}")

            train_process = use_train(
                config['train'], model, opts, dataloaders, save_folder_name)


            train_process.train()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='setting/abide_RGTNet.yaml', type=str,
                        help='Configuration filename for training the model.')
    parser.add_argument('--repeat_time', default=50, type=int)
    args = parser.parse_args()
    torch.cuda.set_device(2)
    # 控制随机性
    seed = 12344
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    cudnn.deterministic = True

    main(args)
