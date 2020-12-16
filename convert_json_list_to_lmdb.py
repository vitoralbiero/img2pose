import argparse
import os

import lmdb
import msgpack
import numpy as np
from tqdm import tqdm

from utils.json_loader import JsonLoader


def json_list_to_lmdb(args):
    cpu_available = os.cpu_count()
    if args.num_workers > cpu_available:
        args.num_workers = cpu_available

    threed_5_points = np.load(args.threed_5_points)
    threed_68_points = np.load(args.threed_68_points)

    print("Loading dataset from %s" % args.json_list)
    data_loader = JsonLoader(
        args.num_workers,
        args.json_list,
        threed_5_points,
        threed_68_points,
        args.dataset_path,
    )

    name = f"{os.path.split(args.json_list)[1][:-4]}.lmdb"
    lmdb_path = os.path.join(args.dest, name)
    isdir = os.path.isdir(lmdb_path)

    if os.path.isfile(lmdb_path):
        os.remove(lmdb_path)

    print(f"Generate LMDB to {lmdb_path}")

    size = len(data_loader) * 1200 * 1200 * 3
    print(f"LMDB max size: {size}")

    db = lmdb.open(
        lmdb_path,
        subdir=isdir,
        map_size=size * 2,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    print(f"Total number of samples: {len(data_loader)}")

    all_pose_labels = []

    txn = db.begin(write=True)

    total_samples = 0

    for idx, data in tqdm(enumerate(data_loader)):
        image, global_pose_labels, bboxes, pose_labels, landmarks = data[0]

        if len(bboxes) == 0:
            continue

        has_pose = False
        for pose_label in pose_labels:
            if pose_label[0] != -9:
                all_pose_labels.append(pose_label)
                has_pose = True

        if not has_pose:
            continue

        txn.put(
            "{}".format(total_samples).encode("ascii"),
            msgpack.dumps((image, global_pose_labels, bboxes, pose_labels, landmarks)),
        )
        if idx % args.write_frequency == 0:
            print(f"[{idx}/{len(data_loader)}]")
            txn.commit()
            txn = db.begin(write=True)

        total_samples += 1

    print(total_samples)

    txn.commit()
    keys = ["{}".format(k).encode("ascii") for k in range(total_samples)]
    with db.begin(write=True) as txn:
        txn.put(b"__keys__", msgpack.dumps(keys))
        txn.put(b"__len__", msgpack.dumps(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()

    if args.train:
        print("Saving pose mean and std dev.")
        all_pose_labels = np.asarray(all_pose_labels)
        pose_mean = np.mean(all_pose_labels, axis=0)
        pose_stddev = np.std(all_pose_labels, axis=0)

        save_file_path = os.path.join(args.dest, os.path.split(args.json_list)[1][:-4])
        np.save(f"{save_file_path}_pose_mean.npy", pose_mean)
        np.save(f"{save_file_path}_pose_stddev.npy", pose_stddev)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_list",
        type=str,
        required=True,
        help="List of json files that contain frames annotations",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset images",
    )
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument(
        "--write_frequency", help="Frequency to save to file.", type=int, default=5000
    )
    parser.add_argument(
        "--dest", type=str, required=True, help="Path to save the lmdb file."
    )
    parser.add_argument(
        "--train", action="store_true", help="Dataset will be used for training."
    )
    parser.add_argument(
        "--threed_5_points",
        type=str,
        help="Reference 3D points to compute pose.",
        default="./pose_references/reference_3d_5_points_trans.npy",
    )

    parser.add_argument(
        "--threed_68_points",
        type=str,
        help="Reference 3D points to compute pose.",
        default="./pose_references/reference_3d_68_points_trans.npy",
    )

    args = parser.parse_args()

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    return args


if __name__ == "__main__":
    args = parse_args()

    json_list_to_lmdb(args)
