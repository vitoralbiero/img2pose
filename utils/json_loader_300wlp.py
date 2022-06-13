import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from .image_operations import bbox_is_dict, expand_bbox_rectangle
from .pose_operations import get_pose, pose_full_image_to_bbox


class FramesJsonList(ImageFolder):
    def __init__(self, threed_5_points, threed_68_points, json_list, dataset_path=None):
        self.samples = []
        self.bboxes = []
        self.landmarks = []
        self.poses_para = []
        self.threed_5_points = threed_5_points
        self.threed_68_points = threed_68_points
        self.dataset_path = dataset_path

        image_paths = pd.read_csv(json_list, delimiter=",", header=None)
        image_paths = np.asarray(image_paths).squeeze()

        print("Loading frames paths...")
        for image_path in tqdm(image_paths):
            with open(image_path) as f:
                image_json = json.load(f)

            # path to the image
            img_path = image_json["image_path"]
            # if not absolute path, append the dataset path
            if self.dataset_path is not None:
                img_path = os.path.join(self.dataset_path, img_path)
            self.samples.append(img_path)

            # landmarks used to create pose labels
            self.landmarks.append(image_json["landmarks"])

            # load bboxes
            self.bboxes.append(image_json["bboxes"])

            # load bboxes
            self.poses_para.append(image_json["pose_para"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path = Path(self.samples[index])

        img = Image.open(image_path)

        (img_w, img_h) = img.size
        global_intrinsics = np.array(
            [[img_w + img_h, 0, img_w // 2], [0, img_w + img_h, img_h // 2], [0, 0, 1]]
        )
        bboxes = self.bboxes[index]
        landmarks = self.landmarks[index]
        pose_para = self.poses_para[index]

        bbox_labels = []
        landmark_labels = []
        pose_labels = []
        global_pose_labels = []

        for i in range(len(bboxes)):
            bbox = np.asarray(bboxes[i])[:4].astype(int)
            landmark = np.asarray(landmarks[i])[:, :2].astype(float)
            pose_para = np.asarray(pose_para[i])[:3].astype(float)

            # remove samples that do not have height ot width or are negative
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                continue

            if -1 in landmark:
                global_pose_labels.append([-9, -9, -9, -9, -9, -9])
                pose_labels.append([-9, -9, -9, -9, -9, -9])

            else:
                P, pose = get_pose(
                    self.threed_68_points,
                    landmark,
                    global_intrinsics,
                )

                pose[:3] = self.convert_aflw(pose_para)
                global_pose_labels.append(pose.tolist())

                projected_bbox = expand_bbox_rectangle(
                    img_w, img_h, 1.1, 1.1, landmark, roll=pose[2]
                )

                local_pose = pose_full_image_to_bbox(
                    pose,
                    global_intrinsics,
                    bbox_is_dict(projected_bbox),
                )

                pose_labels.append(local_pose.tolist())

            bbox_labels.append(projected_bbox.tolist())
            landmark_labels.append(self.landmarks[index][i])

        with open(image_path, "rb") as f:
            raw_img = f.read()

        return (
            raw_img,
            global_pose_labels,
            bbox_labels,
            pose_labels,
            landmark_labels,
        )

    def convert_aflw(self, angle):
        rot_mat_1 = Rotation.from_euler(
            "xyz", [angle[0], -angle[1], -angle[2]], degrees=False
        ).as_matrix()
        rot_mat_2 = np.transpose(rot_mat_1)
        return Rotation.from_matrix(rot_mat_2).as_rotvec()


class JsonLoader(DataLoader):
    def __init__(
        self, workers, json_list, threed_5_points, threed_68_points, dataset_path=None
    ):
        self._dataset = FramesJsonList(
            threed_5_points, threed_68_points, json_list, dataset_path
        )

        super(JsonLoader, self).__init__(
            self._dataset, num_workers=workers, collate_fn=lambda x: x
        )
