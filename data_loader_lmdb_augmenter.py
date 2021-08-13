from os import path

import lmdb
import msgpack
import numpy as np
import six
import torch
from PIL import Image
from torch.utils.data import BatchSampler, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

import utils.augmentation as augmentation
from utils.image_operations import expand_bbox_rectangle
from utils.pose_operations import (get_pose, plot_3d_landmark,
                                   pose_bbox_to_full_image)


class LMDB(Dataset):
    def __init__(
        self,
        config,
        db_path,
        transform=None,
        pose_label_transform=None,
        augmentation_methods=None,
    ):
        self.config = config

        self.env = lmdb.open(
            db_path,
            subdir=path.isdir(db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        with self.env.begin(write=False) as txn:
            self.length = msgpack.loads(txn.get(b"__len__"))
            self.keys = msgpack.loads(txn.get(b"__keys__"))

        self.transform = transform
        self.pose_label_transform = pose_label_transform
        self.augmentation_methods = augmentation_methods

        self.threed_5_points = np.load(self.config.threed_5_points)
        self.threed_68_points = np.load(self.config.threed_68_points)

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        data = msgpack.loads(byteflow)

        # load image
        imgbuf = data[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        # load landmarks label
        landmark_labels = data[4]

        # load bbox
        bbox_labels = np.asarray(data[2])

        # apply augmentations that are provided from the parent class in creation order
        for augmentation_method in self.augmentation_methods:
            img, bbox_labels, landmark_labels = augmentation_method(
                img, bbox_labels, landmark_labels
            )

        # create global intrinsics
        (img_w, img_h) = img.size
        global_intrinsics = np.array(
            [[img_w + img_h, 0, img_w // 2], [0, img_w + img_h, img_h // 2], [0, 0, 1]]
        )

        projected_bbox_labels = []
        pose_labels = []

        img = np.array(img)

        # get pose labels
        for i in range(len(bbox_labels)):
            bbox = bbox_labels[i]
            lms = np.asarray(landmark_labels[i])

            # black out faces that do not have pose annotation
            if -1 in lms:
                img[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2]), :] = 0
                continue

            # convert landmarks to bbox
            bbox_lms = lms.copy()
            bbox_lms[:, 0] -= bbox[0]
            bbox_lms[:, 1] -= bbox[1]

            # create bbox intrinsincs
            w = int(bbox[2] - bbox[0])
            h = int(bbox[3] - bbox[1])

            bbox_intrinsics = np.array(
                [[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]]
            )

            # get pose between gt points and 3D reference
            if len(bbox_lms) == 5:
                P, pose = get_pose(self.threed_5_points, bbox_lms, bbox_intrinsics)
            else:
                P, pose = get_pose(self.threed_68_points, bbox_lms, bbox_intrinsics)

            # convert to global image
            pose_label = pose_bbox_to_full_image(pose, global_intrinsics, bbox)

            # project points and get bbox
            projected_lms, _ = plot_3d_landmark(
                self.threed_68_points, pose_label, global_intrinsics
            )
            projected_bbox = expand_bbox_rectangle(
                img_w, img_h, 1.1, 1.1, projected_lms, roll=pose_label[2]
            )

            pose_labels.append(pose_label)
            projected_bbox_labels.append(projected_bbox)

        pose_labels = np.asarray(pose_labels)

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        target = {
            "dofs": torch.from_numpy(pose_labels).float(),
            "boxes": torch.from_numpy(np.asarray(projected_bbox_labels)).float(),
            "labels": torch.ones((len(bbox_labels),), dtype=torch.int64),
        }

        return img, target

    def __len__(self):
        return self.length


class LMDBDataLoaderAugmenter(DataLoader):
    def __init__(self, config, lmdb_path, train=True):
        self.config = config

        transform = transforms.Compose([transforms.ToTensor()])

        augmentation_methods = []

        if train:
            if self.config.random_flip:
                augmentation_methods.append(augmentation.random_flip)

            if self.config.random_crop:
                augmentation_methods.append(augmentation.random_crop)

            if self.config.noise_augmentation:
                augmentation_methods.append(augmentation.add_noise)

            if self.config.contrast_augmentation:
                augmentation_methods.append(augmentation.change_contrast)

        if self.config.pose_mean is not None:
            pose_label_transform = self.normalize_pose_labels
        else:
            pose_label_transform = None

        self._dataset = LMDB(
            config,
            lmdb_path,
            transform,
            pose_label_transform,
            augmentation_methods,
        )

        if config.distributed:
            self._sampler = DistributedSampler(self._dataset, shuffle=False)

            if train:
                self._sampler = BatchSampler(
                    self._sampler, config.batch_size, drop_last=True
                )

                super(LMDBDataLoaderAugmenter, self).__init__(
                    self._dataset,
                    batch_sampler=self._sampler,
                    pin_memory=config.pin_memory,
                    num_workers=config.workers,
                    collate_fn=collate_fn,
                )
            else:
                super(LMDBDataLoaderAugmenter, self).__init__(
                    self._dataset,
                    config.batch_size,
                    drop_last=False,
                    sampler=self._sampler,
                    pin_memory=config.pin_memory,
                    num_workers=config.workers,
                    collate_fn=collate_fn,
                )

        else:
            super(LMDBDataLoaderAugmenter, self).__init__(
                self._dataset,
                batch_size=config.batch_size,
                shuffle=train,
                pin_memory=config.pin_memory,
                num_workers=config.workers,
                drop_last=True,
                collate_fn=collate_fn,
            )

    def normalize_pose_labels(self, pose_labels):
        for i in range(len(pose_labels)):
            pose_labels[i] = (
                pose_labels[i] - self.config.pose_mean
            ) / self.config.pose_stddev

        return pose_labels


def collate_fn(batch):
    return tuple(zip(*batch))
