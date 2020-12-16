import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from .image_operations import bbox_is_dict
from .pose_operations import get_pose, pose_bbox_to_full_image


class FramesJsonList(ImageFolder):
    def __init__(self, threed_5_points, threed_68_points, json_list, dataset_path=None):
        self.samples = []
        self.bboxes = []
        self.landmarks = []
        self.threed_5_points = threed_5_points
        self.threed_68_points = threed_68_points
        self.dataset_path = dataset_path

        image_paths = pd.read_csv(json_list, delimiter=" ", header=None)
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path = Path(self.samples[index])

        img = Image.open(image_path)

        (w, h) = img.size
        global_intrinsics = np.array(
            [[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]]
        )
        bboxes = self.bboxes[index]
        landmarks = self.landmarks[index]

        bbox_labels = []
        landmark_labels = []
        pose_labels = []
        global_pose_labels = []

        for i in range(len(bboxes)):
            bbox = np.asarray(bboxes[i])[:4].astype(int)
            landmark = np.asarray(landmarks[i])[:, :2].astype(float)

            # remove samples that do not have height ot width or are negative
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                continue

            if -1 in landmark:
                global_pose_labels.append([-9, -9, -9, -9, -9, -9])
                pose_labels.append([-9, -9, -9, -9, -9, -9])

            else:
                landmark[:, 0] -= bbox[0]
                landmark[:, 1] -= bbox[1]

                w = int(bbox[2] - bbox[0])
                h = int(bbox[3] - bbox[1])

                bbox_intrinsics = np.array(
                    [[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]]
                )

                if len(landmark) == 5:
                    P, pose = get_pose(self.threed_5_points, landmark, bbox_intrinsics)
                else:
                    P, pose = get_pose(
                        self.threed_68_points,
                        landmark,
                        bbox_intrinsics,
                    )

                pose_labels.append(pose.tolist())

                global_pose = pose_bbox_to_full_image(
                    pose, global_intrinsics, bbox_is_dict(bbox)
                )

                global_pose_labels.append(global_pose.tolist())

            bbox_labels.append(bbox.tolist())
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
