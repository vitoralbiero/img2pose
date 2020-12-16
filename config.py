import os

import numpy as np
import torch
from easydict import EasyDict
from torch.nn import MSELoss


class Config(EasyDict):
    def __init__(self, args):
        # workspace configuration
        self.prefix = args.prefix
        self.work_path = os.path.join(args.workspace, self.prefix)
        self.model_path = os.path.join(self.work_path, "models")
        try:
            self.create_path(self.model_path)
        except Exception as e:
            print(e)

        self.log_path = os.path.join(self.work_path, "log")
        try:
            self.create_path(self.log_path)
        except Exception as e:
            print(e)

        self.frequency_log = 20

        # training/validation configuration
        self.train_source = args.train_source
        self.val_source = args.val_source

        # network and training parameters
        self.pose_loss = MSELoss(reduction="sum")
        self.pose_mean = np.load(args.pose_mean)
        self.pose_stddev = np.load(args.pose_stddev)
        self.depth = args.depth
        self.lr = args.lr
        self.lr_plateau = args.lr_plateau
        self.early_stop = args.early_stop
        self.batch_size = args.batch_size
        self.workers = args.workers
        self.epochs = args.epochs
        self.min_size = args.min_size
        self.max_size = args.max_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.pin_memory = True

        # resume from or load pretrained weights
        self.pretrained_path = args.pretrained_path
        self.resume_path = args.resume_path

        # online augmentation
        self.noise_augmentation = args.noise_augmentation
        self.contrast_augmentation = args.contrast_augmentation
        self.random_flip = args.random_flip
        self.random_crop = args.random_crop

        # 3d reference points to compute pose
        self.threed_5_points = args.threed_5_points
        self.threed_68_points = args.threed_68_points

        # distributed
        self.distributed = args.distributed
        if not args.distributed:
            self.gpu = 0
        else:
            self.gpu = args.gpu

    def create_path(self, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
