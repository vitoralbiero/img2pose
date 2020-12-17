import torch
from torch.nn import DataParallel, Module
from torch.nn.parallel import DistributedDataParallel
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from model_loader import load_model
from models import FasterDoFRCNN


class WrappedModel(Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module

    def forward(self, images, targets=None):
        return self.module(images, targets)


class img2poseModel:
    def __init__(
        self,
        depth,
        min_size,
        max_size,
        model_path=None,
        device=None,
        pose_mean=None,
        pose_stddev=None,
        distributed=False,
        gpu=0,
        threed_68_points=None,
        threed_5_points=None,
        rpn_pre_nms_top_n_test=6000,
        rpn_post_nms_top_n_test=1000,
        bbox_x_factor=1.1,
        bbox_y_factor=1.1,
        expand_forehead=0.3,
    ):
        self.depth = depth
        self.min_size = min_size
        self.max_size = max_size
        self.model_path = model_path
        self.distributed = distributed
        self.gpu = gpu

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # create network backbone
        backbone = resnet_fpn_backbone(f"resnet{self.depth}", pretrained=True)

        if pose_mean is not None:
            pose_mean = torch.tensor(pose_mean)
            pose_stddev = torch.tensor(pose_stddev)

        if threed_68_points is not None:
            threed_68_points = torch.tensor(threed_68_points)

        if threed_5_points is not None:
            threed_5_points = torch.tensor(threed_5_points)

        # create the feature pyramid network
        self.fpn_model = FasterDoFRCNN(
            backbone,
            2,
            min_size=self.min_size,
            max_size=self.max_size,
            pose_mean=pose_mean,
            pose_stddev=pose_stddev,
            threed_68_points=threed_68_points,
            threed_5_points=threed_5_points,
            rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
            bbox_x_factor=bbox_x_factor,
            bbox_y_factor=bbox_y_factor,
            expand_forehead=expand_forehead,
        )

        # if using cpu, remove the parallel modules from the saved model
        self.fpn_model_without_ddp = self.fpn_model

        if self.distributed:
            self.fpn_model = self.fpn_model.to(self.device)
            self.fpn_model = DistributedDataParallel(
                self.fpn_model, device_ids=[self.gpu]
            )
            self.fpn_model_without_ddp = self.fpn_model.module

            print("Model will use distributed mode!")

        elif str(self.device) == "cpu":
            self.fpn_model = WrappedModel(self.fpn_model)
            self.fpn_model_without_ddp = self.fpn_model

            print("Model will run on CPU!")

        else:
            self.fpn_model = DataParallel(self.fpn_model)
            self.fpn_model = self.fpn_model.to(self.device)
            self.fpn_model_without_ddp = self.fpn_model

            print(f"Model will use {torch.cuda.device_count()} GPUs!")

        if self.model_path is not None:
            self.load_saved_model(self.model_path)
            self.evaluate()

    def load_saved_model(self, model_path):
        load_model(
            self.fpn_model_without_ddp, model_path, cpu_mode=str(self.device) == "cpu"
        )

    def evaluate(self):
        self.fpn_model.eval()

    def train(self):
        self.fpn_model.train()

    def run_model(self, imgs, targets=None):
        outputs = self.fpn_model(imgs, targets)

        return outputs

    def forward(self, imgs, targets):
        losses = self.run_model(imgs, targets)

        return losses

    def predict(self, imgs):
        assert self.fpn_model.training is False

        with torch.no_grad():
            predictions = self.run_model(imgs)

        return predictions
