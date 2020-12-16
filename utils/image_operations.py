import numpy as np
import torch
from PIL import Image


def expand_bbox_rectangle(
    w, h, bbox_x_factor=2.0, bbox_y_factor=2.0, lms=None, expand_forehead=0.3, roll=0
):
    # get a good bbox for the facial landmarks
    min_pt_x = np.min(lms[:, 0], axis=0)
    max_pt_x = np.max(lms[:, 0], axis=0)

    min_pt_y = np.min(lms[:, 1], axis=0)
    max_pt_y = np.max(lms[:, 1], axis=0)

    # find out the bbox of the crop region
    bbox_size_x = int(np.max(max_pt_x - min_pt_x) * bbox_x_factor)
    center_pt_x = 0.5 * min_pt_x + 0.5 * max_pt_x

    bbox_size_y = int(np.max(max_pt_y - min_pt_y) * bbox_y_factor)
    center_pt_y = 0.5 * min_pt_y + 0.5 * max_pt_y

    bbox_min_x, bbox_max_x = (
        center_pt_x - bbox_size_x * 0.5,
        center_pt_x + bbox_size_x * 0.5,
    )

    bbox_min_y, bbox_max_y = (
        center_pt_y - bbox_size_y * 0.5,
        center_pt_y + bbox_size_y * 0.5,
    )

    if abs(roll) > 2.5:
        expand_forehead_size = expand_forehead * np.max(max_pt_y - min_pt_y)
        bbox_max_y += expand_forehead_size

    elif roll > 1:
        expand_forehead_size = expand_forehead * np.max(max_pt_x - min_pt_x)
        bbox_max_x += expand_forehead_size

    elif roll < -1:
        expand_forehead_size = expand_forehead * np.max(max_pt_x - min_pt_x)
        bbox_min_x -= expand_forehead_size

    else:
        expand_forehead_size = expand_forehead * np.max(max_pt_y - min_pt_y)
        bbox_min_y -= expand_forehead_size

    bbox_min_x = bbox_min_x.astype(np.int32)
    bbox_max_x = bbox_max_x.astype(np.int32)
    bbox_min_y = bbox_min_y.astype(np.int32)
    bbox_max_y = bbox_max_y.astype(np.int32)

    # compute necessary padding
    padding_left = abs(min(bbox_min_x, 0))
    padding_top = abs(min(bbox_min_y, 0))
    padding_right = max(bbox_max_x - w, 0)
    padding_bottom = max(bbox_max_y - h, 0)

    # crop the image properly by computing proper crop bounds
    crop_left = 0 if padding_left > 0 else bbox_min_x
    crop_top = 0 if padding_top > 0 else bbox_min_y
    crop_right = w if padding_right > 0 else bbox_max_x
    crop_bottom = h if padding_bottom > 0 else bbox_max_y

    return np.array([crop_left, crop_top, crop_right, crop_bottom])


def expand_bbox_rectangle_tensor(
    w, h, bbox_x_factor=2.0, bbox_y_factor=2.0, lms=None, expand_forehead=0.3, roll=0
):
    # get a good bbox for the facial landmarks
    min_pt_x = torch.min(lms[:, 0], axis=0)[0]
    max_pt_x = torch.max(lms[:, 0], axis=0)[0]

    min_pt_y = torch.min(lms[:, 1], axis=0)[0]
    max_pt_y = torch.max(lms[:, 1], axis=0)[0]

    # find out the bbox of the crop region
    bbox_size_x = int(torch.max(max_pt_x - min_pt_x) * bbox_x_factor)
    center_pt_x = 0.5 * min_pt_x + 0.5 * max_pt_x

    bbox_size_y = int(torch.max(max_pt_y - min_pt_y) * bbox_y_factor)
    center_pt_y = 0.5 * min_pt_y + 0.5 * max_pt_y

    bbox_min_x, bbox_max_x = (
        center_pt_x - bbox_size_x * 0.5,
        center_pt_x + bbox_size_x * 0.5,
    )

    bbox_min_y, bbox_max_y = (
        center_pt_y - bbox_size_y * 0.5,
        center_pt_y + bbox_size_y * 0.5,
    )

    if abs(roll) > 2.5:
        expand_forehead_size = expand_forehead * torch.max(max_pt_y - min_pt_y)
        bbox_max_y += expand_forehead_size

    elif roll > 1:
        expand_forehead_size = expand_forehead * torch.max(max_pt_x - min_pt_x)
        bbox_max_x += expand_forehead_size

    elif roll < -1:
        expand_forehead_size = expand_forehead * torch.max(max_pt_x - min_pt_x)
        bbox_min_x -= expand_forehead_size

    else:
        expand_forehead_size = expand_forehead * torch.max(max_pt_y - min_pt_y)
        bbox_min_y -= expand_forehead_size

    bbox_min_x = bbox_min_x.int()
    bbox_max_x = bbox_max_x.int()
    bbox_min_y = bbox_min_y.int()
    bbox_max_y = bbox_max_y.int()

    # compute necessary padding
    padding_left = abs(min(bbox_min_x, 0))
    padding_top = abs(min(bbox_min_y, 0))
    padding_right = max(bbox_max_x - w, 0)
    padding_bottom = max(bbox_max_y - h, 0)

    # crop the image properly by computing proper crop bounds
    crop_left = 0 if padding_left > 0 else bbox_min_x
    crop_top = 0 if padding_top > 0 else bbox_min_y
    crop_right = w if padding_right > 0 else bbox_max_x
    crop_bottom = h if padding_bottom > 0 else bbox_max_y

    return (
        torch.tensor([crop_left, crop_top, crop_right, crop_bottom])
        .float()
        .to(lms.device)
    )


def preprocess_image_wrt_face(img, bbox_size_factor=2.0, lms=None):
    w, h = img.size

    if lms is None:
        return None, None

    # get a good bbox for the facial landmarks
    min_pt = np.min(lms, axis=0)
    max_pt = np.max(lms, axis=0)

    # find out the bbox of the crop region
    bbox_size = int(np.max(max_pt - min_pt) * bbox_size_factor)
    center_pt = 0.5 * min_pt + 0.5 * max_pt
    bbox_min, bbox_max = center_pt - bbox_size * 0.5, center_pt + bbox_size * 0.5
    bbox_min = bbox_min.astype(np.int32)
    bbox_max = bbox_max.astype(np.int32)

    # compute necessary padding
    padding_left = abs(min(bbox_min[0], 0))
    padding_top = abs(min(bbox_min[1], 0))
    padding_right = max(bbox_max[0] - w, 0)
    padding_bottom = max(bbox_max[1] - h, 0)

    # crop the image properly by computing proper crop bounds
    crop_left = 0 if padding_left > 0 else bbox_min[0]
    crop_top = 0 if padding_top > 0 else bbox_min[1]
    crop_right = w if padding_right > 0 else bbox_max[0]
    crop_bottom = h if padding_bottom > 0 else bbox_max[1]

    cropped_image = img.crop((crop_left, crop_top, crop_right, crop_bottom))

    # copy the cropped image to padded image
    padded_image = Image.new(img.mode, (bbox_size, bbox_size))
    padded_image.paste(
        cropped_image,
        (
            padding_left,
            padding_top,
            padding_left + crop_right - crop_left,
            padding_top + crop_bottom - crop_top,
        ),
    )

    bbox = {}
    bbox["left"] = crop_left - padding_left
    bbox["right"] = crop_right + padding_right
    bbox["top"] = crop_top - padding_top
    bbox["bottom"] = crop_bottom + padding_bottom

    return padded_image, lms, bbox


def bbox_is_dict(bbox):
    # check if the bbox is a not dict and convert it if needed
    if not isinstance(bbox, dict):
        temp_bbox = {}
        temp_bbox["left"] = bbox[0]
        temp_bbox["top"] = bbox[1]
        temp_bbox["right"] = bbox[2]
        temp_bbox["bottom"] = bbox[3]
        bbox = temp_bbox

    return bbox


def pad_image_no_crop(img, bbox):
    bbox = bbox_is_dict(bbox)

    w, h = img.size

    # checks if the bbox is going outside of the image and if so expands the image
    if bbox["left"] < 0 or bbox["top"] < 0 or bbox["right"] > w or bbox["bottom"] > h:
        padding_left = abs(min(bbox["left"], 0))
        padding_top = abs(min(bbox["top"], 0))
        padding_right = max(bbox["right"] - w, 0)
        padding_bottom = max(bbox["bottom"] - h, 0)

        height = h + padding_top + padding_bottom
        width = w + padding_left + padding_right

        padded_image = Image.new(img.mode, (width, height))
        padded_image.paste(
            img, (padding_left, padding_top, padding_left + w, padding_top + h)
        )

        img = padded_image
        bbox["left"] += padding_left
        bbox["top"] += padding_top
        bbox["right"] += padding_left
        bbox["bottom"] += padding_top

    return img, bbox


def crop_face_bbox_expanded(img, bbox, bbox_size_factor=2.0):
    bbox = bbox_is_dict(bbox)

    # get image size
    w, h = img.size

    # transform bounding box to 8 points (4,2)
    x1, y1, x2, y2 = bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]
    bounding_box_8pts = np.array([x1, y1, x2, y1, x2, y2, x1, y2])
    bounding_box_8pts = bounding_box_8pts.reshape(-1, 2)

    # get a good bbox for the facial landmarks
    min_pt = np.min(bounding_box_8pts, axis=0)
    max_pt = np.max(bounding_box_8pts, axis=0)

    # find out the bbox of the crop region
    bbox_size = int(np.max(max_pt - min_pt) * bbox_size_factor)
    center_pt = 0.5 * min_pt + 0.5 * max_pt
    bbox_min, bbox_max = center_pt - bbox_size * 0.5, center_pt + bbox_size * 0.5
    bbox_min = bbox_min.astype(np.int32)
    bbox_max = bbox_max.astype(np.int32)

    # compute necessary padding
    padding_left = abs(min(bbox_min[0], 0))
    padding_top = abs(min(bbox_min[1], 0))
    padding_right = max(bbox_max[0] - w, 0)
    padding_bottom = max(bbox_max[1] - h, 0)

    # crop the image properly by computing proper crop bounds
    crop_left = 0 if padding_left > 0 else bbox_min[0]
    crop_top = 0 if padding_top > 0 else bbox_min[1]
    crop_right = w if padding_right > 0 else bbox_max[0]
    crop_bottom = h if padding_bottom > 0 else bbox_max[1]

    cropped_image = img.crop((crop_left, crop_top, crop_right, crop_bottom))

    # copy the cropped image to padded image
    padded_image = Image.new(img.mode, (bbox_size, bbox_size))
    padded_image.paste(
        cropped_image,
        (
            padding_left,
            padding_top,
            padding_left + crop_right - crop_left,
            padding_top + crop_bottom - crop_top,
        ),
    )

    bbox_padded = {}
    bbox_padded["left"] = crop_left - padding_left
    bbox_padded["right"] = crop_right + padding_right
    bbox_padded["top"] = crop_top - padding_top
    bbox_padded["bottom"] = crop_bottom + padding_bottom

    bbox = {}
    bbox["left"] = crop_left
    bbox["right"] = crop_right
    bbox["top"] = crop_top
    bbox["bottom"] = crop_bottom

    return padded_image, bbox_padded, bbox


def resize_image(img, min_size=600, max_size=1000):
    width = img.width
    height = img.height
    w, h, scale = width, height, 1.0

    if width < height:
        if width < min_size:
            w = min_size
            h = int(height * min_size / width)
            scale = float(min_size) / float(width)
        elif width > max_size:
            w = max_size
            h = int(height * max_size / width)
            scale = float(max_size) / float(width)
    else:
        if height < min_size:
            w = int(width * min_size / height)
            h = min_size
            scale = float(min_size) / float(height)
        elif height > max_size:
            w = int(width * max_size / height)
            h = max_size
            scale = float(max_size) / float(height)

    img_resized = img.resize((w, h))
    img_resize_info = [h, w, scale]

    return img_resized, img_resize_info
