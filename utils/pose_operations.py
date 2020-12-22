import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from .face_align import norm_crop
from .image_operations import bbox_is_dict, expand_bbox_rectangle


def bbox_dict_to_np(bbox):
    bbox_np = np.zeros(shape=4)
    bbox_np[0] = bbox["left"]
    bbox_np[1] = bbox["top"]
    bbox_np[2] = bbox["right"]
    bbox_np[3] = bbox["bottom"]

    return bbox_np


def quat_to_rotation_mat_tensor(quat):
    x = quat[0]
    y = quat[1]
    z = quat[2]
    w = quat[3]
    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w
    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w
    matrix = torch.zeros(3, 3).to(quat.device)
    matrix[0, 0] = x2 - y2 - z2 + w2
    matrix[1, 0] = 2 * (xy + zw)
    matrix[2, 0] = 2 * (xz - yw)
    matrix[0, 1] = 2 * (xy - zw)
    matrix[1, 1] = -x2 + y2 - z2 + w2
    matrix[2, 1] = 2 * (yz + xw)
    matrix[0, 2] = 2 * (xz + yw)
    matrix[1, 2] = 2 * (yz - xw)
    matrix[2, 2] = -x2 - y2 + z2 + w2
    return matrix


def from_rotvec_tensor(rotvec):
    norm = torch.norm(rotvec)
    small_angle = norm <= 1e-3
    scale = 0
    if small_angle:
        scale = 0.5 - norm ** 2 / 48 + norm ** 4 / 3840
    else:
        scale = torch.sin(norm / 2) / norm
    quat = torch.zeros(4).to(rotvec.device)
    quat[0:3] = scale * rotvec
    quat[3] = torch.cos(norm / 2)

    return quat_to_rotation_mat_tensor(quat)


def transform_points_tensor(points, pose):
    return torch.matmul(points, from_rotvec_tensor(pose[:3]).T) + pose[3:]


def get_bbox_intrinsics(image_intrinsics, bbox):
    # crop principle point of view
    bbox_center_x = bbox["left"] + ((bbox["right"] - bbox["left"]) // 2)
    bbox_center_y = bbox["top"] + ((bbox["bottom"] - bbox["top"]) // 2)

    # create a camera intrinsics from the bbox center
    bbox_intrinsics = image_intrinsics.copy()
    bbox_intrinsics[0, 2] = bbox_center_x
    bbox_intrinsics[1, 2] = bbox_center_y

    return bbox_intrinsics


def get_bbox_intrinsics_np(image_intrinsics, bbox):
    # crop principle point of view
    bbox_center_x = bbox[0] + ((bbox[2] - bbox[0]) // 2)
    bbox_center_y = bbox[1] + ((bbox[3] - bbox[1]) // 2)

    # create a camera intrinsics from the bbox center
    bbox_intrinsics = image_intrinsics.copy()
    bbox_intrinsics[0, 2] = bbox_center_x
    bbox_intrinsics[1, 2] = bbox_center_y

    return bbox_intrinsics


def pose_full_image_to_bbox(pose, image_intrinsics, bbox):
    # check if bbox is np or dict
    bbox = bbox_is_dict(bbox)

    # rotation vector
    rvec = pose[:3].copy()

    # translation and scale vector
    tvec = pose[3:].copy()

    # get camera intrinsics using bbox
    bbox_intrinsics = get_bbox_intrinsics(image_intrinsics, bbox)

    # focal length
    focal_length = image_intrinsics[0, 0]

    # bbox_size
    bbox_width = bbox["right"] - bbox["left"]
    bbox_height = bbox["bottom"] - bbox["top"]
    bbox_size = bbox_width + bbox_height

    # project crop points using the full image camera intrinsics
    projected_point = image_intrinsics.dot(tvec.T)

    # reverse the projected points using the crop camera intrinsics
    tvec = projected_point.dot(np.linalg.inv(bbox_intrinsics.T))

    # adjust scale
    tvec[2] /= focal_length / bbox_size

    # same for rotation
    rmat = Rotation.from_rotvec(rvec).as_matrix()
    # project crop points using the crop camera intrinsics
    projected_point = image_intrinsics.dot(rmat)
    # reverse the projected points using the full image camera intrinsics
    rmat = np.linalg.inv(bbox_intrinsics).dot(projected_point)
    rvec = Rotation.from_matrix(rmat).as_rotvec()

    return np.concatenate([rvec, tvec])


def pose_bbox_to_full_image(pose, image_intrinsics, bbox):
    # check if bbox is np or dict
    bbox = bbox_is_dict(bbox)

    # rotation vector
    rvec = pose[:3].copy()

    # translation and scale vector
    tvec = pose[3:].copy()

    # get camera intrinsics using bbox
    bbox_intrinsics = get_bbox_intrinsics(image_intrinsics, bbox)

    # focal length
    focal_length = image_intrinsics[0, 0]

    # bbox_size
    bbox_width = bbox["right"] - bbox["left"]
    bbox_height = bbox["bottom"] - bbox["top"]
    bbox_size = bbox_width + bbox_height

    # adjust scale
    tvec[2] *= focal_length / bbox_size

    # project crop points using the crop camera intrinsics
    projected_point = bbox_intrinsics.dot(tvec.T)

    # reverse the projected points using the full image camera intrinsics
    tvec = projected_point.dot(np.linalg.inv(image_intrinsics.T))

    # same for rotation
    rmat = Rotation.from_rotvec(rvec).as_matrix()
    # project crop points using the crop camera intrinsics
    projected_point = bbox_intrinsics.dot(rmat)
    # reverse the projected points using the full image camera intrinsics
    rmat = np.linalg.inv(image_intrinsics).dot(projected_point)
    rvec = Rotation.from_matrix(rmat).as_rotvec()

    return np.concatenate([rvec, tvec])


def plot_3d_landmark(verts, campose, intrinsics):
    lm_3d_trans = transform_points(verts, campose)

    # project to image plane
    lms_3d_trans_proj = intrinsics.dot(lm_3d_trans.T).T
    lms_projected = (
        lms_3d_trans_proj[:, :2] / np.tile(lms_3d_trans_proj[:, 2], (2, 1)).T
    )

    return lms_projected, lms_3d_trans_proj


def plot_3d_landmark_torch(verts, campose, intrinsics):
    lm_3d_trans = transform_points_tensor(verts, campose)

    # project to image plane
    lms_3d_trans_proj = torch.matmul(intrinsics, lm_3d_trans.T).T
    lms_projected = lms_3d_trans_proj[:, :2] / lms_3d_trans_proj[:, 2].repeat(2, 1).T

    return lms_projected


def transform_points(points, pose):
    return points.dot(Rotation.from_rotvec(pose[:3]).as_matrix().T) + pose[3:]


def get_pose(vertices, twod_landmarks, camera_intrinsics, initial_pose=None):
    threed_landmarks = vertices
    twod_landmarks = np.asarray(twod_landmarks).astype("float32")

    # if initial_pose is provided, use it as a guess to solve new pose
    if initial_pose is not None:
        initial_pose = np.asarray(initial_pose)
        retval, rvecs, tvecs = cv2.solvePnP(
            threed_landmarks,
            twod_landmarks,
            camera_intrinsics,
            None,
            rvec=initial_pose[:3],
            tvec=initial_pose[3:],
            flags=cv2.SOLVEPNP_EPNP,
            useExtrinsicGuess=True,
        )
    else:
        retval, rvecs, tvecs = cv2.solvePnP(
            threed_landmarks,
            twod_landmarks,
            camera_intrinsics,
            None,
            flags=cv2.SOLVEPNP_EPNP,
        )

    rotation_mat = np.zeros(shape=(3, 3))
    R = cv2.Rodrigues(rvecs, rotation_mat)[0]

    RT = np.column_stack((R, tvecs))
    P = np.matmul(camera_intrinsics, RT)
    dof = np.append(rvecs, tvecs)

    return P, dof


def transform_pose_global_project_bbox(
    boxes,
    dofs,
    pose_mean,
    pose_stddev,
    image_shape,
    threed_68_points=None,
    bbox_x_factor=1.1,
    bbox_y_factor=1.1,
    expand_forehead=0.3,
):
    if len(dofs) == 0:
        return boxes, dofs

    device = dofs.device

    boxes = boxes.cpu().numpy()
    dofs = dofs.cpu().numpy()

    threed_68_points = threed_68_points.numpy()

    (h, w) = image_shape
    global_intrinsics = np.array([[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]])

    if threed_68_points is not None:
        threed_68_points = threed_68_points

    pose_mean = pose_mean.numpy()
    pose_stddev = pose_stddev.numpy()

    dof_mean = pose_mean
    dof_std = pose_stddev
    dofs = dofs * dof_std + dof_mean

    projected_boxes = []
    global_dofs = []

    for i in range(dofs.shape[0]):
        global_dof = pose_bbox_to_full_image(dofs[i], global_intrinsics, boxes[i])
        global_dofs.append(global_dof)

        if threed_68_points is not None:
            # project points and get bbox
            projected_lms, _ = plot_3d_landmark(
                threed_68_points, global_dof, global_intrinsics
            )
            projected_bbox = expand_bbox_rectangle(
                w,
                h,
                bbox_x_factor=bbox_x_factor,
                bbox_y_factor=bbox_y_factor,
                lms=projected_lms,
                roll=global_dof[2],
                expand_forehead=expand_forehead,
            )
        else:
            projected_bbox = boxes[i]

        projected_boxes.append(projected_bbox)

    global_dofs = torch.from_numpy(np.asarray(global_dofs)).float()
    projected_boxes = torch.from_numpy(np.asarray(projected_boxes)).float()

    return projected_boxes.to(device), global_dofs.to(device)


def align_faces(threed_5_points, img, poses, face_size=224):
    if len(poses) == 0:
        return None
    elif np.ndim(poses) == 1:
        poses = poses[np.newaxis, :]

    (w, h) = img.size
    global_intrinsics = np.array([[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]])

    faces_aligned = []

    for pose in poses:
        proj_lms, _ = plot_3d_landmark(
            threed_5_points, np.asarray(pose), global_intrinsics
        )
        face_aligned = norm_crop(np.asarray(img).copy(), proj_lms, face_size)
        faces_aligned.append(Image.fromarray(face_aligned))

    return faces_aligned
