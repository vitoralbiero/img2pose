import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


def random_crop(img, bboxes, landmarks):
    useable_landmarks = False
    for landmark in landmarks:
        if -1 not in landmark:
            useable_landmarks = True
            break

    total_attempts = 10
    searching = True
    attempt = 0

    while searching:
        crop_img = img.copy()
        (w, h) = crop_img.size
        crop_size = random.uniform(0.7, 1)

        if attempt == total_attempts:
            return img, bboxes, landmarks

        crop_x = int(w * crop_size)
        crop_y = int(h * crop_size)

        start_x = random.randint(0, w - crop_x)
        start_y = (start_x // w) * h

        crop_bbox = [start_x, start_y, start_x + crop_x, start_y + crop_y]

        new_bboxes, new_lms = _adjust_bboxes_landmarks(
            bboxes.copy(), landmarks.copy(), crop_bbox
        )

        if len(new_bboxes) > 0:
            if useable_landmarks:
                for lms in new_lms:
                    if -1 not in lms:
                        searching = False
                        break
            else:
                searching = False

        if not searching:
            crop_img = crop_img.crop(
                (crop_bbox[0], crop_bbox[1], crop_bbox[2], crop_bbox[3])
            )

        attempt += 1

    return crop_img, new_bboxes, new_lms


def _adjust_bboxes_landmarks(bboxes, landmarks, crop_bbox):
    new_bboxes = []
    new_lms = []
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        lms = np.asarray(landmarks[i])

        bbox_center_x = bbox[0] + ((bbox[2] - bbox[0]) // 2)
        bbox_center_y = bbox[1] + ((bbox[3] - bbox[1]) // 2)

        if (
            bbox_center_x > crop_bbox[0]
            and bbox_center_x < crop_bbox[2]
            and bbox_center_y > crop_bbox[1]
            and bbox_center_y < crop_bbox[3]
        ):
            bbox[[0, 2]] -= crop_bbox[0]
            bbox[[1, 3]] -= crop_bbox[1]

            bbox[0] = max(bbox[0], 0)
            bbox[1] = max(bbox[1], 0)
            bbox[2] = min(bbox[2], crop_bbox[2] - crop_bbox[0])
            bbox[3] = min(bbox[3], crop_bbox[3] - crop_bbox[1])

            add_lm = True
            for lm in lms:
                if (
                    lm[0] < crop_bbox[0]
                    or lm[0] > crop_bbox[2]
                    or lm[1] < crop_bbox[1]
                    or lm[1] > crop_bbox[3]
                ):
                    add_lm = False
                    break

            if add_lm:
                lms[:, 0] -= crop_bbox[0]
                lms[:, 1] -= crop_bbox[1]

                new_lms.append(lms.tolist())
                new_bboxes.append(bbox)

    return new_bboxes, new_lms


def random_flip(img, bboxes, all_landmarks):
    flip = random.randint(0, 1)

    if flip == 1:
        # flip image
        img = ImageOps.mirror(img)

        # flip bboxes
        old_bboxes = bboxes.copy()
        (w, h) = img.size
        bboxes[:, 0] = w - old_bboxes[:, 2]
        bboxes[:, 2] = w - old_bboxes[:, 0]

        for i in range(len(all_landmarks)):
            landmarks = np.asarray(all_landmarks[i])

            if -1 in landmarks:
                continue

            if len(landmarks) == 5:
                order = [1, 0, 2, 4, 3]
            else:
                order = [
                    16,
                    15,
                    14,
                    13,
                    12,
                    11,
                    10,
                    9,
                    8,
                    7,
                    6,
                    5,
                    4,
                    3,
                    2,
                    1,
                    0,
                    26,
                    25,
                    24,
                    23,
                    22,
                    21,
                    20,
                    19,
                    18,
                    17,
                    27,
                    28,
                    29,
                    30,
                    35,
                    34,
                    33,
                    32,
                    31,
                    45,
                    44,
                    43,
                    42,
                    47,
                    46,
                    39,
                    38,
                    37,
                    36,
                    41,
                    40,
                    54,
                    53,
                    52,
                    51,
                    50,
                    49,
                    48,
                    59,
                    58,
                    57,
                    56,
                    55,
                    64,
                    63,
                    62,
                    61,
                    60,
                    67,
                    66,
                    65,
                ]

            # flip landmarks
            landmarks[:, 0] = w - landmarks[:, 0]
            flandmarks = landmarks.copy()
            for idx, a in enumerate(order):
                flandmarks[idx, :] = landmarks[a, :]

            all_landmarks[i] = flandmarks.tolist()

    return img, bboxes, all_landmarks


def rotate(img, landmarks, bbox):
    angle = random.gauss(0, 1) * 30

    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # Transform Image
    new_img = _rotate_img(img, angle, cX, cY, h, w)

    # Transform Landmarks
    new_landmarks = _rotate_landmarks(landmarks, angle, cX, cY, h, w)

    # Transform Bounding Box
    x1, y1, x2, y2 = bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]
    bounding_box_8pts = np.array([x1, y1, x2, y1, x2, y2, x1, y2])

    rot_bounding_box_8pts = _rotate_bbox(bounding_box_8pts, angle, cX, cY, h, w)
    rot_bounding_box = _get_enclosing_bbox(rot_bounding_box_8pts, y2 - y1, x2 - x1)[
        0
    ].astype("float")

    new_bbox = {
        "left": rot_bounding_box[0],
        "top": rot_bounding_box[1],
        "right": rot_bounding_box[2],
        "bottom": rot_bounding_box[3],
    }

    # validate that bbox boundaries are within the image otherwise do not apply rotation
    if (
        new_bbox["top"] > 0
        and new_bbox["bottom"] < h
        and bbox["left"] > 0
        and bbox["right"] < w
    ):
        img = new_img
        landmarks = new_landmarks
        bbox = new_bbox

    return img, landmarks, bbox


def scale(img, bbox):
    scale = random.uniform(0.75, 1.25)
    bbox = _scale_bbox(img, bbox, scale)

    return bbox


def translate_vertical(img, bbox):
    bbox_height = bbox["bottom"] - bbox["top"]
    vtrans = random.uniform(-0.1, 0.1) * bbox_height

    # check if bbox boundaries are within image, otherwise do not move bbox
    if bbox["top"] + vtrans > 0 and bbox["bottom"] + vtrans < img.shape[0]:
        bbox["top"] += vtrans
        bbox["bottom"] += vtrans

    return bbox


def translate_horizontal(img, bbox):
    bbox_width = bbox["right"] - bbox["left"]
    htrans = random.uniform(-0.1, 0.1) * bbox_width

    # check if bbox boundaries are within image, otherwise do not move bbox
    if bbox["left"] + htrans > 0 and bbox["right"] + htrans < img.shape[1]:
        bbox["left"] += htrans
        bbox["right"] += htrans

    return bbox


def change_contrast(img, bboxes, landmarks):
    change = random.randint(0, 1)
    if change == 1:
        factor = random.uniform(0.5, 1.5)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(factor)

    return img, bboxes, landmarks


def add_noise(img, bboxes, landmarks):
    add_noise = random.randint(0, 4)
    if add_noise == 4:
        noise_types = ["gauss", "s&p", "poisson"]
        noise_idx = random.randint(0, 2)
        noise_type = noise_types[noise_idx]

        img = np.array(img)

        if noise_type == "gauss":
            row, col, ch = img.shape
            mean = 0
            var = 0.1
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            img = img + gauss

        elif noise_type == "s&p":
            row, col, ch = img.shape
            s_vs_p = 0.5
            amount = 0.004
            # Salt mode
            num_salt = np.ceil(amount * (img.size / ch) * s_vs_p)
            coords = [
                np.random.randint(0, i - 1, int(num_salt)) for i in img.shape[0:2]
            ]
            img[tuple(coords)] = 255
            # Pepper mode
            num_pepper = np.ceil(amount * (img.size / ch) * (1.0 - s_vs_p))
            coords = [
                np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape[0:2]
            ]
            img[tuple(coords)] = 0

        elif noise_type == "poisson":
            vals = len(np.unique(img))
            vals = 2 ** np.ceil(np.log2(vals))
            img = np.random.poisson(img * vals) / float(vals)

        img = Image.fromarray(img.astype("uint8"))

    return img, bboxes, landmarks


def _rotate_img(img, angle, cX, cY, h, w):
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    img = cv2.warpAffine(img, M, (nW, nH))

    return img


def _rotate_landmarks(landmarks, angle, cx, cy, h, w):
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy

    landmarks = np.append(landmarks, np.ones((landmarks.shape[0], 1)), axis=1)
    calculated = (np.dot(M, landmarks.T)).T
    return calculated.astype("int")


def _rotate_bbox(corners, angle, cx, cy, h, w):
    corners = corners.reshape(-1, 2)
    corners = np.hstack(
        (corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0])))
    )
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M, corners.T).T
    calculated = calculated.reshape(-1, 8)

    return calculated


def _get_enclosing_bbox(corners, original_height, original_width):
    x = corners[:, [0, 2, 4, 6]]
    y = corners[:, [1, 3, 5, 7]]
    xmin = np.min(x, 1).reshape(-1, 1)
    ymin = np.min(y, 1).reshape(-1, 1)
    xmax = np.max(x, 1).reshape(-1, 1)
    ymax = np.max(y, 1).reshape(-1, 1)

    height = ymax - ymin
    width = xmax - xmin

    diff_height = height - original_height
    diff_width = width - original_width

    ymax -= diff_height // 2
    ymin += diff_height // 2
    xmax -= diff_width // 2
    xmin += diff_width // 2

    final = np.hstack((xmin, ymin, xmax, ymax, corners[:, 8:]))

    return final.astype("int")


def _scale_bbox(img, bbox, scale):
    height = bbox["bottom"] - bbox["top"]
    width = bbox["right"] - bbox["left"]

    new_height = height * scale
    new_width = width * scale

    diff_height = height - new_height
    diff_width = width - new_width

    # check if bbox boundaries are within image, otherwise do not scale bbox
    if (
        bbox["top"] + (diff_height // 2) > 0
        and bbox["bottom"] - (diff_height // 2) < img.shape[0]
        and bbox["left"] + (diff_width // 2) > 0
        and bbox["right"] - (diff_width // 2) < img.shape[1]
    ):
        bbox["bottom"] -= diff_height // 2
        bbox["top"] += diff_height // 2
        bbox["right"] -= diff_width // 2
        bbox["left"] += diff_width // 2

    return bbox
