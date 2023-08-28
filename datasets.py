"""Datasets"""

import os

import torch
from torch.utils.data import DataLoader, Dataset, random_split, SubsetRandomSampler, Subset
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision
import glob
import PIL
import random
import math
import pickle
import numpy as np
import torch.nn.functional as F
import imageio


class CelebA(Dataset):
    """CelelebA Dataset"""

    def __init__(self, dataset_path, img_size, **kwargs):
        super().__init__()

        self.data = glob.glob(dataset_path)
        assert (
            len(self.data) > 0
        ), "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
            [
                transforms.Resize(320),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((img_size, img_size), interpolation=0),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)

        return X, 0


class Cats(Dataset):
    """Cats Dataset"""

    def __init__(self, dataset_path, img_size, **kwargs):
        super().__init__()

        self.data = glob.glob(dataset_path)
        assert (
            len(self.data) > 0
        ), "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), interpolation=0),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)

        return X, 0


class Carla(Dataset):
    """Carla Dataset"""

    # dataset_poses_path = '/home/shreya/pi-GAN/carla_poses/*.npy'
    def __init__(self, dataset_path, dataset_poses_path, img_size, **kwargs):
        super().__init__()

        self.data_imgs = glob.glob(dataset_path)
        self.data_poses = glob.glob(dataset_poses_path)
        self.data_imgs.sort()
        self.data_poses.sort()
        self.flip_y_z = np.array(
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
            ],
            dtype=np.float16,  # Make cuda mixed precision happy
        )

        assert (
            len(self.data_imgs) > 0
        ), "Can't find data; make sure you specify the path to your dataset"
        assert (
            len(self.data_poses) > 0
        ), "Can't find image poses; make sure you specify the path to your dataset for poses"
        assert len(self.data_imgs) == len(
            self.data_poses
        ), "Number of images and poses do not match"

        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), interpolation=0),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.data_imgs)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data_imgs[index])
        X = self.transform(X)
        ex_pose = self.flip_y_z @ np.load(self.data_poses[index])

        return X, 0, ex_pose




class Cats(Dataset):
    """Cats Dataset"""

    def __init__(self, dataset_path, img_size, **kwargs):
        super().__init__()

        self.data = glob.glob(dataset_path)
        assert (
            len(self.data) > 0
        ), "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), interpolation=0),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)

        return X, 0


# class Car_SRN(Dataset):
#     """Car SRN Dataset"""

#     # dataset_path =  "/home/shreya/3d-learning-project/srn_cars/cars_train/"
#     def __init__(self, dataset_path, img_size, **kwargs):
#         super().__init__()

#         self.data_img_folders = glob.glob(dataset_path)
#         # self.data_pose_folders = glob.glob(dataset_poses_path)
#         self.data_img_folders.sort()
#         # self.data_poses.sort()
#         # self.flip_y_z = np.array(
#         #     [
#         #         [1, 0, 0],
#         #         [0, 0, 1],
#         #         [0, 1, 0],
#         #     ],
#         #     dtype=np.float16,  # Make cuda mixed precision happy
#         # )

#         assert (
#             len(self.data_img_folders) > 0
#         ), "Can't find data; make sure you specify the path to your dataset"
#         # assert (
#         #     len(self.data_poses) > 0
#         # ), "Can't find image poses; make sure you specify the path to your dataset for poses"
#         assert len(self.data_img_folders) == len(
#             self.data_poses
#         ), "Number of images and poses do not match"

#         self.transform = transforms.Compose(
#             [
#                 transforms.Resize((img_size, img_size), interpolation=0),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.5], [0.5]),
#             ]
#         )

#     def __len__(self):
#         return len(self.data_imgs)

#     def __getitem__(self, index):
#         img_path = self.data_img_folders[index]
        
#         X = PIL.Image.open(self.data_imgs[index])
#         X = self.transform(X)
#         ex_pose = self.flip_y_z @ np.load(self.data_poses[index])

#         return X, 0, ex_pose

def get_image_to_tensor_balanced(image_size=0):
    ops = []
    if image_size > 0:
        ops.append(transforms.Resize(image_size))
    ops.extend(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
    )
    return transforms.Compose(ops)


def get_mask_to_tensor():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))]
    )

# https://github.com/sxyu/pixel-nerf/blob/master/src/data/SRNDataset.py
class Car_SRN(torch.utils.data.Dataset):
    """
    Dataset from SRN (V. Sitzmann et al. 2020)
    """
    # path = "/home/shreya/3d-learning-project/srn_cars/cars"
    def __init__(
        self, dataset_path, stage="train", img_size=128, world_scale=1.0, **kwargs
    ):
        """
        :param stage train | val | test
        :param image_size result image size (resizes if different)
        :param world_scale amount to scale entire world by
        """
        path = dataset_path
        image_size = (img_size, img_size)
        super().__init__()
        self.base_path = path + "_" + stage
        self.dataset_name = os.path.basename(path)

        self.only_upper_hem = kwargs["only_upper_hem"]

        print("Loading SRN dataset", self.base_path, "name:", self.dataset_name)
        self.stage = stage
        assert os.path.exists(self.base_path)

        is_chair = "chair" in self.dataset_name
        if is_chair and stage == "train":
            # Ugly thing from SRN's public dataset
            tmp = os.path.join(self.base_path, "chairs_2.0_train")
            if os.path.exists(tmp):
                self.base_path = tmp

        self.intrins = sorted(
            glob.glob(os.path.join(self.base_path, "*", "intrinsics.txt"))
        )
        self.image_to_tensor = get_image_to_tensor_balanced()
        self.mask_to_tensor = get_mask_to_tensor()

        self.image_size = image_size
        self.world_scale = world_scale
        self._coord_trans = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        )
        self.flip_y_z = torch.tensor(
            [
                [1.0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ],
        )

        if is_chair:
            self.z_near = 1.25
            self.z_far = 2.75
        else:
            self.z_near = 0.8
            self.z_far = 1.8
        self.lindisp = False

    def __len__(self):
        return len(self.intrins)

    def __getitem__(self, index):
        intrin_path = self.intrins[index]
        dir_path = os.path.dirname(intrin_path)
        rgb_paths = sorted(glob.glob(os.path.join(dir_path, "rgb", "*")))
        pose_paths = sorted(glob.glob(os.path.join(dir_path, "pose", "*")))

        assert len(rgb_paths) == len(pose_paths)

        with open(intrin_path, "r") as intrinfile:
            lines = intrinfile.readlines()
            focal, cx, cy, _ = map(float, lines[0].split())
            height, width = map(int, lines[-1].split())

        all_imgs = []
        all_poses = []
        all_pose_org = []
        all_masks = []
        all_bboxes = []
        for rgb_path, pose_path in zip(rgb_paths, pose_paths):
            img = imageio.imread(rgb_path)[..., :3]
            img_tensor = self.image_to_tensor(img)
            mask = (img != 255).all(axis=-1)[..., None].astype(np.uint8) * 255
            mask_tensor = self.mask_to_tensor(mask)

            pose = torch.from_numpy(
                np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4)
            )
            # print("Pose org - ", pose)
            pose_org = pose.clone()
            pose = pose @ self._coord_trans
            pose = self.flip_y_z @ pose
            # print("Pose org - ", pose)
            
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rnz = np.where(rows)[0]
            cnz = np.where(cols)[0]
            if len(rnz) == 0:
                raise RuntimeError(
                    "ERROR: Bad image at", rgb_path, "please investigate!"
                )
            rmin, rmax = rnz[[0, -1]]
            cmin, cmax = cnz[[0, -1]]
            bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)

            all_imgs.append(img_tensor)
            all_masks.append(mask_tensor)
            all_poses.append(pose)
            all_pose_org.append(pose_org)
            all_bboxes.append(bbox)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        all_pose_org = torch.stack(all_pose_org)
        all_masks = torch.stack(all_masks)
        all_bboxes = torch.stack(all_bboxes)

        if all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size[0] / all_imgs.shape[-2]
            focal *= scale
            cx *= scale
            cy *= scale
            all_bboxes *= scale

            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")

        if self.world_scale != 1.0:
            focal *= self.world_scale
            all_poses[:, :3, 3] *= self.world_scale
        focal = torch.tensor(focal, dtype=torch.float32)

        if self.only_upper_hem:
            is_upper = all_pose_org[:,2,3] > 0
            all_poses = all_poses[is_upper]
            all_imgs = all_imgs[is_upper]


        all_poses = all_poses[..., :3, :4]

        perm = torch.randperm(all_poses.size(0))
        idx = perm[:3]

        result = {
            "path": dir_path,
            "img_id": index,
            "focal": focal,
            "c": torch.tensor([cx, cy], dtype=torch.float32),
            "images": all_imgs,
            "masks": all_masks,
            "bbox": all_bboxes,
            "poses": all_poses,
        }
        return all_imgs[idx], 0, all_poses[idx]

def get_dataset(name, subsample=None, batch_size=1, **kwargs):
    dataset = globals()[name](**kwargs)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=False,
        num_workers=8,
    )
    return dataloader, 0


def get_dataset_distributed(name, world_size, rank, batch_size, enable_split=False, **kwargs):
    dataset = globals()[name](**kwargs)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
    )

    if not enable_split:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            num_workers=4,
        )

        return dataloader, 3

    else:
        train_size = 0.8

        dataset_size = len(dataset)
        train_len = int(train_size * dataset_size)
        test_len = dataset_size - train_len
        # train_data, test_data = random_split(dataset, [train_len, test_len])
        train_data = Subset(dataset, range(train_len))
        test_data = Subset(dataset, range(train_len, dataset_size))

        # train_sampler = SubsetRandomSampler(train_data.indices)
        # test_sampler = SubsetRandomSampler(test_data.indices)

        # train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        # test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_data, test_data, 3
