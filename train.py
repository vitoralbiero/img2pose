import argparse
import random
from os import path

import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from config import Config
from data_loader_lmdb import LMDBDataLoader
from data_loader_lmdb_augmenter import LMDBDataLoaderAugmenter
from early_stop import EarlyStop
from img2pose import img2poseModel
from model_loader import load_model, save_model
from train_logger import TrainLogger
from utils.dist import init_distributed_mode, is_main_process, reduce_dict


class Train:
    def __init__(self, config):
        self.config = config

        if is_main_process():
            # start tensorboard summary writer
            self.writer = SummaryWriter(config.log_path)

        # load training dataset generator
        if self.config.random_flip or self.config.random_crop:
            self.train_loader = LMDBDataLoaderAugmenter(
                self.config, self.config.train_source
            )
        else:
            self.train_loader = LMDBDataLoader(self.config, self.config.train_source)
        print(f"Training with {len(self.train_loader.dataset)} images.")

        # loads validation dataset generator if a validation dataset is given
        if self.config.val_source is not None:
            self.val_loader = LMDBDataLoader(self.config, self.config.val_source, False)

        # creates model
        self.img2pose_model = img2poseModel(
            depth=self.config.depth,
            min_size=self.config.min_size,
            max_size=self.config.max_size,
            device=self.config.device,
            pose_mean=self.config.pose_mean,
            pose_stddev=self.config.pose_stddev,
            distributed=self.config.distributed,
            gpu=self.config.gpu,
            threed_68_points=np.load(self.config.threed_68_points),
            threed_5_points=np.load(self.config.threed_5_points),
        )
        # optimizer for the backbone and heads
        if args.optimizer == "Adam":
            self.optimizer = optim.Adam(
                self.img2pose_model.fpn_model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
        elif args.optimizer == "SGD":
            self.optimizer = optim.SGD(
                self.img2pose_model.fpn_model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
                momentum=self.config.momentum,
            )
        else:
            raise Exception("No optimizer founded, please select between SGD or Adam.")

        # loads a model with optimizer so that it can continue training where it stopped
        if self.config.resume_path:
            print(f"Resuming training from {self.config.resume_path}")
            load_model(
                self.img2pose_model.fpn_model,
                self.config.resume_path,
                model_only=False,
                optimizer=self.optimizer,
                cpu_mode=str(self.config.device) == "cpu",
            )

        # loads a pretrained model without loading the optimizer
        if self.config.pretrained_path:
            print(f"Loading pretrained weights from {self.config.pretrained_path}")
            load_model(
                self.img2pose_model.fpn_model,
                self.config.pretrained_path,
                model_only=True,
                cpu_mode=str(self.config.device) == "cpu",
            )

        if is_main_process():
            # saves configuration to file for easier retrival later
            print(self.config)
            self.save_file(self.config, "config.txt")

        if is_main_process():
            # saves optimizer config to file for easier retrival later
            print(self.optimizer)
            self.save_file(self.optimizer, "optimizer.txt")

        self.tensorboard_loss_every = max(len(self.train_loader) // 100, 1)

        # reduce learning rate when the validation loss stops to decrease
        if self.config.lr_plateau:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.1,
                patience=3,
                verbose=True,
                threshold=0.001,
                cooldown=1,
                min_lr=0.00001,
            )

        # stops training before the defined epochs if validation loss stops to decrease
        if self.config.early_stop:
            self.early_stop = EarlyStop(mode="min", patience=5)

    def run(self):
        self.img2pose_model.train()

        # accumulate running loss to log into tensorboard
        running_losses = {}
        running_losses["loss"] = 0

        step = 0

        # prints the best step and loss every time it does a validation
        self.best_step = 0
        self.best_val_loss = float("Inf")

        for epoch in range(self.config.epochs):
            train_logger = TrainLogger(
                self.config.batch_size, self.config.frequency_log, self.config.num_gpus
            )
            idx = 0
            for idx, data in enumerate(self.train_loader):
                imgs, targets = data

                imgs = [image.to(self.config.device) for image in imgs]
                targets = [
                    {k: v.to(self.config.device) for k, v in t.items()} for t in targets
                ]

                self.optimizer.zero_grad()

                # forward pass
                losses = self.img2pose_model.forward(imgs, targets)

                loss = sum(loss for loss in losses.values())

                # does a backward propagation through the network
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.img2pose_model.fpn_model.parameters(), 10
                )

                self.optimizer.step()

                if self.config.distributed:
                    losses = reduce_dict(losses)
                    loss = sum(loss for loss in losses.values())

                for loss_name in losses.keys():
                    if loss_name in running_losses:
                        running_losses[loss_name] += losses[loss_name].item()
                    else:
                        running_losses[loss_name] = losses[loss_name].item()

                running_losses["loss"] += loss.item()

                # saves loss into tensorboard
                if step % self.tensorboard_loss_every == 0 and step != 0:
                    for loss_name in running_losses.keys():
                        if is_main_process():
                            self.writer.add_scalar(
                                f"train_{loss_name}",
                                running_losses[loss_name] / self.tensorboard_loss_every,
                                step,
                            )

                        running_losses[loss_name] = 0

                train_logger(
                    epoch, self.config.epochs, idx, len(self.train_loader), loss.item()
                )
                step += 1

            # evaluate model using validation set (if set)
            if self.config.val_source is not None:
                val_loss = self.evaluate(step)

            else:
                # otherwise just save the model
                save_model(
                    self.img2pose_model.fpn_model_without_ddp,
                    self.optimizer,
                    self.config,
                    step=step,
                )

            # if validation loss stops decreasing, decrease lr
            if self.config.lr_plateau and self.config.val_source is not None:
                self.scheduler.step(val_loss)

            # early stop model to prevent overfitting
            if self.config.early_stop and self.config.val_source is not None:
                self.early_stop(val_loss)
                if self.early_stop.stop:
                    print("Early stopping model...")
                    break

        if self.config.val_source is not None:
            val_loss = self.evaluate(step)

    def checkpoint(self, val_loss, step):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_step = step

            save_model(
                self.img2pose_model.fpn_model_without_ddp,
                self.optimizer,
                self.config,
                val_loss,
                step,
            )

    def reduce_lr(self):
        for params in self.optimizer.param_groups:
            params["lr"] /= 10

        print("Reducing learning rate...")
        print(self.optimizer)

    def evaluate(self, step):
        val_losses = {}
        val_losses["loss"] = 0

        print("Evaluating model...")
        with torch.no_grad():
            for data in iter(self.val_loader):
                imgs, targets = data

                imgs = [image.to(self.config.device) for image in imgs]
                targets = [
                    {k: v.to(self.config.device) for k, v in t.items()} for t in targets
                ]

                if self.config.distributed:
                    torch.cuda.synchronize()

                losses = self.img2pose_model.forward(imgs, targets)

                if self.config.distributed:
                    losses = reduce_dict(losses)

                loss = sum(loss for loss in losses.values())

                for loss_name in losses.keys():
                    if loss_name in val_losses:
                        val_losses[loss_name] += losses[loss_name].item()
                    else:
                        val_losses[loss_name] = losses[loss_name].item()

                val_losses["loss"] += loss.item()

        for loss_name in val_losses.keys():
            if is_main_process():
                self.writer.add_scalar(
                    f"val_{loss_name}",
                    round(val_losses[loss_name] / len(self.val_loader), 6),
                    step,
                )

        val_loss = round(val_losses["loss"] / len(self.val_loader), 6)
        self.checkpoint(val_loss, step)

        print(
            "Current validation loss: "
            + f"{val_loss:.6f} at step {step}"
            + " - Best validation loss: "
            + f"{self.best_val_loss:.6f} at step {self.best_step}"
        )

        self.img2pose_model.train()

        return val_loss

    def save_file(self, string, file_name):
        with open(path.join(self.config.work_path, file_name), "w") as file:
            file.write(str(string))
            file.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a deep network to predict 3D expression and 6DOF pose."
    )
    # network and training parameters
    parser.add_argument(
        "--min_size", help="Min size", default="640, 672, 704, 736, 768, 800", type=str
    )
    parser.add_argument("--max_size", help="Max size", default=1400, type=int)
    parser.add_argument("--epochs", help="Number of epochs.", default=100, type=int)
    parser.add_argument(
        "--depth", help="Number of layers [18, 50 or 101].", default=18, type=int
    )
    parser.add_argument("--lr", help="Learning rate.", default=0.001, type=float)
    parser.add_argument(
        "--optimizer", help="Optimizer (SGD or Adam).", default="SGD", type=str
    )
    parser.add_argument("--batch_size", help="Batch size.", default=2, type=int)
    parser.add_argument(
        "--lr_plateau", help="Reduce lr on plateau.", action="store_true"
    )
    parser.add_argument("--early_stop", help="Use early stop.", action="store_true")
    parser.add_argument("--workers", help="Workers number.", default=4, type=int)
    parser.add_argument(
        "--pose_mean", help="Pose mean file path.", type=str, required=True
    )
    parser.add_argument(
        "--pose_stddev", help="Pose stddev file path.", type=str, required=True
    )

    # training/validation configuration
    parser.add_argument(
        "--workspace", help="Worskspace path to save models and logs.", required=True
    )
    parser.add_argument(
        "--train_source", help="Path to the dataset train LMDB file.", required=True
    )
    parser.add_argument(
        "--val_source", help="Path to the dataset validation LMDB file."
    )

    parser.add_argument(
        "--prefix", help="Prefix to save the model.", type=str, required=True
    )

    # resume from or load pretrained weights
    parser.add_argument(
        "--pretrained_path", help="Path to pretrained weights.", type=str
    )
    parser.add_argument(
        "--resume_path", help="Path to load model to resume training.", type=str
    )

    # online augmentation
    parser.add_argument("--noise_augmentation", action="store_true")
    parser.add_argument("--contrast_augmentation", action="store_true")
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--random_crop", action="store_true")

    # distributed training parameters
    parser.add_argument(
        "--world-size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist-url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--distributed", help="Use distributed training", action="store_true"
    )

    # reference points to create pose labels
    parser.add_argument(
        "--threed_5_points",
        type=str,
        help="Reference 3D points to compute pose.",
        default="./pose_references/reference_3d_5_points_trans.npy",
    )

    parser.add_argument(
        "--threed_68_points",
        type=str,
        help="Reference 3D points to compute pose.",
        default="./pose_references/reference_3d_68_points_trans.npy",
    )

    args = parser.parse_args()
    
    args.min_size = [int(item) for item in args.min_size.split(",")]

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.distributed:
        init_distributed_mode(args)

    config = Config(args)

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    train = Train(config)
    train.run()
