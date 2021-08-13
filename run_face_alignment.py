import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from img2pose import img2poseModel
from model_loader import load_model
from utils.pose_operations import align_faces


class img2pose:
    def __init__(self, args):
        self.threed_5_points = np.load(args.threed_5_points)
        self.threed_68_points = np.load(args.threed_68_points)
        self.nms_threshold = args.nms_threshold

        self.pose_mean = np.load(args.pose_mean)
        self.pose_stddev = np.load(args.pose_stddev)
        self.model = self.create_model(args)

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.min_size = (args.min_size,)
        self.max_size = args.max_size

        self.max_faces = args.max_faces
        self.face_size = args.face_size
        self.order_method = args.order_method
        self.det_threshold = args.det_threshold

        images_path = args.images_path
        if os.path.isfile(images_path):
            self.image_list = pd.read_csv(images_path, delimiter=" ", header=None)
            self.image_list = np.asarray(self.image_list).squeeze()
        else:
            self.image_list = [
                os.path.join(images_path, img_path)
                for img_path in os.listdir(images_path)
            ]

        self.output_path = args.output_path

    def create_model(self, args):
        img2pose_model = img2poseModel(
            args.depth,
            args.min_size,
            args.max_size,
            pose_mean=self.pose_mean,
            pose_stddev=self.pose_stddev,
            threed_68_points=self.threed_68_points,
        )
        load_model(
            img2pose_model.fpn_model,
            args.pretrained_path,
            cpu_mode=str(img2pose_model.device) == "cpu",
            model_only=True,
        )
        img2pose_model.evaluate()

        return img2pose_model

    def align(self):
        for img_path in tqdm(self.image_list):
            image_name = os.path.split(img_path)[-1]
            img = Image.open(img_path).convert("RGB")

            res = self.model.predict([self.transform(img)])[0]

            all_scores = res["scores"].cpu().numpy().astype("float")
            all_poses = res["dofs"].cpu().numpy().astype("float")

            all_poses = all_poses[all_scores > self.det_threshold]
            all_scores = all_scores[all_scores > self.det_threshold]

            if len(all_poses) > 0:
                if self.order_method == "confidence":
                    order = np.argsort(all_scores)[::-1]

                elif self.order_method == "position":
                    distance_center = np.sqrt(
                        all_poses[:, 3] ** 2
                        + all_poses[:, 4] ** 2
                        + all_poses[:, 5] ** 2
                    )

                    order = np.argsort(distance_center)

                top_poses = all_poses[order][: self.max_faces]

                sub_folder = os.path.basename(
                    os.path.normpath(os.path.split(img_path)[0])
                )
                output_path = os.path.join(args.output_path, sub_folder)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                for i in range(len(top_poses)):
                    save_name = image_name
                    if len(top_poses) > 1:
                        name, ext = image_name.split(".")
                        save_name = f"{name}_{i}.{ext}"

                    aligned_face = align_faces(self.threed_5_points, img, top_poses[i])[
                        0
                    ]
                    aligned_face = aligned_face.resize((self.face_size, self.face_size))
                    aligned_face.save(os.path.join(output_path, save_name))
            else:
                print(f"No face detected above the threshold {self.det_threshold}!")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Align top n faces ordering by score or distance to image center."
    )
    parser.add_argument("--max_faces", help="Top n faces to save.", default=1, type=int)
    parser.add_argument(
        "--order_method",
        help="How to order faces [confidence, position].",
        default="position",
        type=str,
    )
    parser.add_argument(
        "--face_size",
        help="Image size to save aligned faces [112 or 224].",
        default=224,
        type=int,
    )
    parser.add_argument("--min_size", help="Image min size", default=400, type=int)
    parser.add_argument("--max_size", help="Image max size", default=1400, type=int)
    parser.add_argument(
        "--depth", help="Number of layers [18, 50 or 101].", default=18, type=int
    )
    parser.add_argument(
        "--pose_mean",
        help="Pose mean file path.",
        type=str,
        default="./models/WIDER_train_pose_mean_v1.npy",
    )
    parser.add_argument(
        "--pose_stddev",
        help="Pose stddev file path.",
        type=str,
        default="./models/WIDER_train_pose_stddev_v1.npy",
    )

    parser.add_argument(
        "--pretrained_path",
        help="Path to pretrained weights.",
        type=str,
        default="./models/img2pose_v1.pth",
    )

    parser.add_argument(
        "--threed_5_points",
        type=str,
        help="Reference 3D points to align the face.",
        default="./pose_references/reference_3d_5_points_trans.npy",
    )

    parser.add_argument(
        "--threed_68_points",
        type=str,
        help="Reference 3D points to project bbox.",
        default="./pose_references/reference_3d_68_points_trans.npy",
    )

    parser.add_argument("--nms_threshold", default=0.6, type=float)
    parser.add_argument(
        "--det_threshold", help="Detection threshold.", default=0.7, type=float
    )
    parser.add_argument("--images_path", help="Image list, or folder.", required=True)
    parser.add_argument("--output_path", help="Path to save predictions", required=True)

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    return args


if __name__ == "__main__":
    args = parse_args()

    img2pose = img2pose(args)
    img2pose.align()
