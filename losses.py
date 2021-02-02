from itertools import chain, repeat

import torch
import torch.nn.functional as F

from utils.pose_operations import plot_3d_landmark_torch, pose_full_image_to_bbox


def fastrcnn_loss(
    class_logits,
    class_labels,
    dof_regression,
    labels,
    dof_regression_targets,
    proposals,
    image_shapes,
    pose_mean=None,
    pose_stddev=None,
    threed_points=None,
):
    # # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Arguments:
        class_logits (Tensor)
        dof_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        dof_loss (Tensor)
        points_loss (Tensor)
    """
    img_size = [
        (boxes_in_image.shape[0], image_shapes[i])
        for i, boxes_in_image in enumerate(proposals)
    ]
    img_size = list(chain.from_iterable(repeat(j, i) for i, j in img_size))

    labels = torch.cat(labels, dim=0)
    class_labels = torch.cat(class_labels, dim=0)
    dof_regression_targets = torch.cat(dof_regression_targets, dim=0)
    proposals = torch.cat(proposals, dim=0)
    classification_loss = F.cross_entropy(class_logits, class_labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    N = dof_regression.shape[0]
    dof_regression = dof_regression.reshape(N, -1, 6)
    dof_regression = dof_regression[sampled_pos_inds_subset, labels_pos]
    prop_regression = proposals[sampled_pos_inds_subset]

    dof_regression_targets = dof_regression_targets[sampled_pos_inds_subset]
    
    all_target_calibration_points = None
    all_pred_calibration_points = None

    for i in range(prop_regression.shape[0]):
        (h, w) = img_size[i]
        global_intrinsics = torch.Tensor(
            [[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]]
        ).to(proposals[0].device)

        threed_points = threed_points.to(proposals[0].device)

        h = prop_regression[i, 3] - prop_regression[i, 1]
        w = prop_regression[i, 2] - prop_regression[i, 0]
        local_intrinsics = torch.Tensor(
            [[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]]
        ).to(proposals[0].device)

        # calibration points projection
        local_dof_regression = (
            dof_regression[i, :] * pose_stddev.to(proposals[0].device)
        ) + pose_mean.to(proposals[0].device)

        pred_calibration_points = plot_3d_landmark_torch(
            threed_points, local_dof_regression.float(), local_intrinsics
        ).unsqueeze(0)

        # pose convertion for pose loss
        dof_regression_targets[i, :] = torch.from_numpy(
            pose_full_image_to_bbox(
                dof_regression_targets[i, :].cpu().numpy(),
                global_intrinsics.cpu().numpy(),
                prop_regression[i, :].cpu().numpy(),
            )
        ).to(proposals[0].device)

        # target calibration points projection
        target_calibration_points = plot_3d_landmark_torch(
            threed_points, dof_regression_targets[i, :], local_intrinsics
        ).unsqueeze(0)

        if all_target_calibration_points is None:
            all_target_calibration_points = target_calibration_points
        else:
            all_target_calibration_points = torch.cat(
                (all_target_calibration_points, target_calibration_points)
            )
        if all_pred_calibration_points is None:
            all_pred_calibration_points = pred_calibration_points
        else:
            all_pred_calibration_points = torch.cat(
                (all_pred_calibration_points, pred_calibration_points)
            )

        if pose_mean is not None:
            dof_regression_targets[i, :] = (
                dof_regression_targets[i, :] - pose_mean.to(proposals[0].device)
            ) / pose_stddev.to(proposals[0].device)

    points_loss = F.l1_loss(all_target_calibration_points, all_pred_calibration_points)

    dof_loss = (
        F.mse_loss(
            dof_regression,
            dof_regression_targets,
            reduction="sum",
        )
        / dof_regression.shape[0]
    )

    return classification_loss, dof_loss, points_loss
