import argparse
import os
import sys
import time

import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
from tqdm import tqdm

sys.path.append("./")
from img2pose import img2poseModel
from model_loader import load_model


class WIDER_Eval:
    def __init__(self, args):
        self.threed_68_points = np.load(args.threed_68_points)
        self.nms_threshold = args.nms_threshold
        self.pose_mean = np.load(args.pose_mean)
        self.pose_stddev = np.load(args.pose_stddev)

        self.test_dataset = self.get_dataset(args)
        self.dataset_path = args.dataset_path
        self.img2pose_model = self.create_model(args)

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.min_size = args.min_size
        self.max_size = args.max_size
        self.flip = len(args.min_size) > 1
        self.output_path = args.output_path

    def create_model(self, args):
        img2pose_model = img2poseModel(
            args.depth,
            args.min_size[-1],
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

    def get_dataset(self, args):
        annotations = open(args.dataset_list)
        lines = annotations.readlines()

        test_dataset = []

        for i in range(len(lines)):
            lines[i] = str(lines[i].rstrip("\n"))
            if "--" in lines[i]:
                test_dataset.append(lines[i])

        return test_dataset

    def bbox_voting(self, bboxes, iou_thresh=0.6):
        # bboxes: a numpy array of N*5 size representing N boxes;
        #         for each box, it is represented as [x1, y1, x2, y2, s]
        # iou_thresh: group bounding boxes if their overlap is > threshold.

        order = bboxes[:, 4].ravel().argsort()[::-1]
        bboxes = bboxes[order, :]
        areas = (bboxes[:, 2] - bboxes[:, 0] + 1) * (bboxes[:, 3] - bboxes[:, 1] + 1)
        voted_bboxes = np.zeros([0, 5])
        while bboxes.shape[0] > 0:
            xx1 = np.maximum(bboxes[0, 0], bboxes[:, 0])
            yy1 = np.maximum(bboxes[0, 1], bboxes[:, 1])
            xx2 = np.minimum(bboxes[0, 2], bboxes[:, 2])
            yy2 = np.minimum(bboxes[0, 3], bboxes[:, 3])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            overlaps = inter / (areas[0] + areas[:] - inter)
            merge_indexs = np.where(overlaps >= iou_thresh)[0]
            if merge_indexs.shape[0] == 0:
                bboxes = np.delete(bboxes, np.array([0]), 0)
                areas = np.delete(areas, np.array([0]), 0)
                continue
            bboxes_accu = bboxes[merge_indexs, :]
            bboxes = np.delete(bboxes, merge_indexs, 0)
            areas = np.delete(areas, merge_indexs, 0)
            # generate a new box by score voting and box voting
            bbox = np.zeros((1, 5))
            box_weights = (bboxes_accu[:, -1] / max(bboxes_accu[:, -1])) * overlaps[
                merge_indexs
            ]
            bboxes_accu[:, 0:4] = bboxes_accu[:, 0:4] * np.tile(
                box_weights.reshape((-1, 1)), (1, 4)
            )
            bbox[:, 0:4] = np.sum(bboxes_accu[:, 0:4], axis=0) / (np.sum(box_weights))
            bbox[0, 4] = np.sum(bboxes_accu[:, 4] * box_weights)
            voted_bboxes = np.row_stack((voted_bboxes, bbox))

        return voted_bboxes

    def get_scales(self, im):
        im_shape = im.size
        im_size_min = np.min(im_shape[0:2])

        scales = [float(scale) / im_size_min for scale in self.min_size]

        return scales

    def test(self):
        times = []

        for img_path in tqdm(self.test_dataset):
            img_full_path = os.path.join(self.dataset_path, img_path)
            img = Image.open(img_full_path).convert("RGB")

            bboxes = []

            (w, h) = img.size
            scales = self.get_scales(img)

            if self.flip:
                flip_list = (False, True)
            else:
                flip_list = (False,)

            for flip in flip_list:
                for scale in scales:
                    run_img = img.copy()

                    if flip:
                        run_img = ImageOps.mirror(run_img)

                    new_w = int(run_img.size[0] * scale)
                    new_h = int(run_img.size[1] * scale)

                    min_size = min(new_w, new_h)
                    max_size = max(new_w, new_h)

                    if len(scales) > 1:
                        self.img2pose_model.fpn_model.module.set_max_min_size(
                            max_size, min_size
                        )

                    time1 = time.time()

                    res = self.img2pose_model.predict([self.transform(run_img)])

                    time2 = time.time()
                    times.append(time2 - time1)

                    res = res[0]

                    for i in range(len(res["scores"])):
                        bbox = res["boxes"].cpu().numpy()[i].astype("int")
                        score = res["scores"].cpu().numpy()[i]
                        pose = res["dofs"].cpu().numpy()[i]

                        if flip:
                            bbox_copy = bbox.copy()
                            bbox[0] = w - bbox_copy[2]
                            bbox[2] = w - bbox_copy[0]
                            pose[1:4] *= -1

                        bboxes.append(np.append(bbox, score))

            bboxes = np.asarray(bboxes)

            if len(self.min_size) > 1:
                bboxes = self.bbox_voting(bboxes, self.nms_threshold)

            if np.ndim(bboxes) == 1 and len(bboxes) > 0:
                bboxes = bboxes[np.newaxis, :]

            output_path = os.path.join(self.output_path, os.path.split(img_path)[0])

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            file_name = os.path.split(img_path)[-1]
            f = open(os.path.join(output_path, file_name[:-4] + ".txt"), "w")
            f.write(file_name + "\n")
            f.write(str(len(bboxes)) + "\n")
            for i in range(len(bboxes)):
                bbox = bboxes[i]

                f.write(
                    f"{bbox[0]} {bbox[1]} {bbox[2]-bbox[0]} {bbox[3]-bbox[1]} {bbox[4]}\n"
                )
            f.close()

        print(f"Average time forward pass: {np.mean(np.asarray(times))}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a deep network to predict 3D expression and 6DOF pose."
    )
    parser.add_argument(
        "--min_size",
        help="Min size",
        default="200, 300, 500, 800, 1100, 1400, 1700",
        type=str,
    )
    parser.add_argument("--max_size", help="Max size", default=1400, type=int)
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

    # training/validation configuration
    parser.add_argument("--output_path", help="Path to save predictions", required=True)
    parser.add_argument("--dataset_path", help="Path to the dataset", required=True)
    parser.add_argument("--dataset_list", help="Dataset list.")

    # resume from or load pretrained weights
    parser.add_argument(
        "--pretrained_path", help="Path to pretrained weights.", type=str
    )
    parser.add_argument("--nms_threshold", default=0.6, type=float)

    parser.add_argument(
        "--threed_68_points",
        type=str,
        help="Reference 3D points to compute pose.",
        default="./pose_references/reference_3d_68_points_trans.npy",
    )

    args = parser.parse_args()

    args.min_size = [int(item) for item in args.min_size.split(", ")]

    return args


if __name__ == "__main__":
    args = parse_args()

    wider_eval = WIDER_Eval(args)
    wider_eval.test()
