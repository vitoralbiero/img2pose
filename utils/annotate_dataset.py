import argparse
import json
import sys

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# modify to your RetinaFace path
sys.path.append("../insightface/RetinaFace/")
import os

from retinaface import RetinaFace


def face_landmark_detection(detector, args):
    image_paths = pd.read_csv(args.image_list, delimiter=" ", header=None)
    image_paths = np.asarray(image_paths).squeeze()

    for i in tqdm(range(len(image_paths))):
        img_json = {}

        img_path = image_paths[i]
        img_name = os.path.split(img_path)[-1]

        output_path = os.path.join(args.output_path, os.path.split(img_name)[0])
        file_output_path = os.path.join(
            output_path, f"{os.path.split(img_name)[-1][:-4]}.json"
        )

        if os.path.isfile(file_output_path):
            print(f"Skipping file {img_name} as it was already processed.")
            continue

        im = cv2.imread(img_path)

        pyramid = True
        do_flip = False

        if not pyramid:
            target_size = 1200
            max_size = 1600
            target_size = 1504
            max_size = 2000
            target_size = 1600
            max_size = 2150
            im_shape = im.shape
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])
            im_scale = float(target_size) / float(im_size_min)
            # prevent bigger axis from being more than max_size:
            if np.round(im_scale * im_size_max) > max_size:
                im_scale = float(max_size) / float(im_size_max)
            scales = [im_scale]
        else:
            do_flip = True
            TEST_SCALES = [500, 800, 1100, 1400, 1700]
            target_size = 800
            max_size = 1200
            im_shape = im.shape
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])
            im_scale = float(target_size) / float(im_size_min)
            # prevent bigger axis from being more than max_size:
            if np.round(im_scale * im_size_max) > max_size:
                im_scale = float(max_size) / float(im_size_max)
            scales = [float(scale) / target_size * im_scale for scale in TEST_SCALES]

        faces, landmarks = detector.detect(
            im, threshold=args.thresh, scales=scales, do_flip=do_flip
        )

        bboxes_list = []
        landmarks_list = []

        if faces is not None:
            for i in range(faces.shape[0]):
                bbox = faces[i].astype(np.float32)
                if landmarks is not None:
                    landmark5 = landmarks[i].astype(np.float32)

                    bboxes_list.append(bbox.tolist())
                    landmarks_list.append(landmark5.tolist())

        if len(landmarks_list) > 0:
            img_json["image_path"] = img_path
            img_json["bboxes"] = bboxes_list
            img_json["landmarks"] = landmarks_list

            if not os.path.isdir(output_path):
                os.makedirs(output_path)

            with open(
                file_output_path,
                "w",
            ) as output_file:
                json.dump(img_json, output_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_list",
        type=str,
        required=True,
        help="List with path to images.",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save the json files."
    )
    parser.add_argument(
        "--thresh", type=float, default=0.8, help="Face detection threshold."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../insightface/RetinaFace/models/R50/R50",
        help="Model path for detector.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    return args


if __name__ == "__main__":
    args = parse_args()

    detector = RetinaFace(args.model_path, 0, 0, "net3", vote=False)

    face_landmark_detection(detector, args)
