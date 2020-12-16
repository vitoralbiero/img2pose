import cv2
import numpy as np
from Sim3DR import RenderPipeline

from .pose_operations import plot_3d_landmark


def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order="C")
    return arr


def get_colors(img, ver):
    h, w, _ = img.shape
    ver[0, :] = np.minimum(np.maximum(ver[0, :], 0), w - 1)  # x
    ver[1, :] = np.minimum(np.maximum(ver[1, :], 0), h - 1)  # y
    ind = np.round(ver).astype(np.int32)
    colors = img[ind[1, :], ind[0, :], :] / 255.0  # n x 3

    return colors.copy()


class Renderer:
    def __init__(
        self,
        vertices_path="../pose_references/vertices_trans.npy",
        triangles_path="../pose_references/triangles.npy",
    ):
        self.vertices = np.load(vertices_path)
        self.triangles = _to_ctype(np.load(triangles_path).T)
        self.vertices[:, 0] *= -1

        self.cfg = {
            "intensity_ambient": 0.3,
            "color_ambient": (1, 1, 1),
            "intensity_directional": 0.6,
            "color_directional": (1, 1, 1),
            "intensity_specular": 0.1,
            "specular_exp": 5,
            "light_pos": (0, 0, 5),
            "view_pos": (0, 0, 5),
        }

        self.render_app = RenderPipeline(**self.cfg)

    def transform_vertices(self, img, poses, global_intrinsics=None):
        (w, h) = img.size
        if global_intrinsics is None:
            global_intrinsics = np.array(
                [[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]]
            )

        transformed_vertices = []
        for pose in poses:
            projected_lms = np.zeros_like(self.vertices)
            projected_lms[:, :2], lms_3d_trans_proj = plot_3d_landmark(
                self.vertices, pose, global_intrinsics
            )
            projected_lms[:, 2] = lms_3d_trans_proj[:, 2] * -1

            range_x = np.max(projected_lms[:, 0]) - np.min(projected_lms[:, 0])
            range_y = np.max(projected_lms[:, 1]) - np.min(projected_lms[:, 1])

            s = (h + w) / pose[5]
            projected_lms[:, 2] *= s
            projected_lms[:, 2] += (range_x + range_y) * 3

            transformed_vertices.append(projected_lms)

        return transformed_vertices

    def render(self, img, transformed_vertices, alpha=0.9, save_path=None):
        img = np.asarray(img)
        overlap = img.copy()

        for vertices in transformed_vertices:
            vertices = _to_ctype(vertices)  # transpose
            overlap = self.render_app(vertices, self.triangles, overlap)

        res = cv2.addWeighted(img, 1 - alpha, overlap, alpha, 0)

        if save_path is not None:
            cv2.imwrite(save_path, res)
            print(f"Save visualization result to {save_path}")

        return res

    def save_to_obj(self, img, ver_lst, height, save_path):
        n_obj = len(ver_lst)  # count obj

        if n_obj <= 0:
            return

        n_vertex = ver_lst[0].T.shape[1]
        n_face = self.triangles.shape[0]

        with open(save_path, "w") as f:
            for i in range(n_obj):
                ver = ver_lst[i].T
                colors = get_colors(img, ver)

                for j in range(n_vertex):
                    x, y, z = ver[:, j]
                    f.write(
                        f"v {x:.2f} {height - y:.2f} {z:.2f} {colors[j, 2]:.2f} "
                        f"{colors[j, 1]:.2f} {colors[j, 0]:.2f}\n"
                    )

            for i in range(n_obj):
                offset = i * n_vertex
                for j in range(n_face):
                    idx1, idx2, idx3 = self.triangles[j]  # m x 3
                    f.write(
                        f"f {idx3 + 1 + offset} {idx2 + 1 + offset} "
                        f"{idx1 + 1 + offset}\n"
                    )

        print(f"Dump tp {save_path}")
