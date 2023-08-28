"""Script to render a video using a trained pi-GAN  model."""

import argparse
import math
import os

from torchvision.utils import save_image

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import numpy as np
import skvideo.io
import curriculums

import train
import datasets

from torch_ema import ExponentialMovingAverage

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('--seeds', nargs='+', default=[0, 1, 2])
parser.add_argument('--output_dir', type=str, default='vids')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--max_batch_size', type=int, default=2400000)
parser.add_argument('--depth_map', action='store_true')
parser.add_argument('--lock_view_dependence', action='store_true')
# parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--ray_step_multiplier', type=int, default=1)
# parser.add_argument('--ray_step_multiplier', type=int, default=2)
parser.add_argument('--num_frames', type=int, default=36)
parser.add_argument('--curriculum', type=str, default='CelebA')
parser.add_argument('--trajectory', type=str, default='front')
parser.add_argument('--input_img_num', type=int, default=0)
opt = parser.parse_args()

os.makedirs(opt.output_dir, exist_ok=True)

curriculum = getattr(curriculums, opt.curriculum)
curriculum['num_steps'] = 12
# curriculum[0]['num_steps'] * opt.ray_step_multiplier
curriculum['img_size'] = opt.image_size
curriculum['psi'] = 0.7
curriculum['v_stddev'] = 0
curriculum['h_stddev'] = 0
curriculum['lock_view_dependence'] = opt.lock_view_dependence
curriculum['last_back'] = curriculum.get('eval_last_back', False)
curriculum['num_frames'] = opt.num_frames
curriculum['nerf_noise'] = 0
curriculum = {key: value for key, value in curriculum.items() if type(key) is str}

def tensor_to_PIL(img):
    img = img.squeeze() * 0.5 + 0.5
    return Image.fromarray(img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())

generator = torch.load(opt.path, map_location=device)
ema_file = opt.path.split('generator')[0] + 'ema.pth'

ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
ema.load_state_dict(torch.load(ema_file, map_location=device))

# ema = torch.load(ema_file, map_location=device)
print("ema -", type(ema))
# print(ema.keys())
# dict_keys(['decay', 'num_updates', 'shadow_params', 'collected_params'])
# ~/pi-GAN
ema.copy_to(generator.parameters())
generator.set_device(device)
generator.eval()

if opt.trajectory == 'front':
    trajectory = []
    for t in np.linspace(0, 1, curriculum['num_frames']):
        pitch = 0.2 * np.cos(t * 2 * math.pi) + math.pi/2
        yaw = 0.4 * np.sin(t * 2 * math.pi) + math.pi/2
        fov = 12

        fov = 12 + 5 + np.sin(t * 2 * math.pi) * 5

        trajectory.append((pitch, yaw, fov))
elif opt.trajectory == 'orbit':
    trajectory = []
    for t in np.linspace(0, 1, curriculum['num_frames']):
        pitch = math.pi/4
        yaw = t * 2 * math.pi
        fov = curriculum['fov']

        trajectory.append((pitch, yaw, fov))

print(curriculum)
is_multiview = curriculum["dataset"] == "Car_SRN"

if is_multiview:
    dataset, _, _ = datasets.get_dataset_distributed(curriculum['dataset'], 1, 0, 1, enable_split=True,
        dataset_path=curriculum['dataset_path'],img_size=32, only_upper_hem=is_multiview)
else:
    dataset, _, _ = datasets.get_dataset_distributed("Carla", 1, 0, 1, enable_split=True,
            dataset_path=curriculum['dataset_path'], dataset_poses_path=curriculum['dataset_poses_path'], img_size=32)



only_upper_hem = is_multiview



# img, _, ex_pose = next(iter(dataset))


for img_num in range(5):
    img, _, ex_pose = dataset[img_num]
    if is_multiview:
        img = img[0]
        ex_pose = ex_pose[0]
    # opt.input_img_num
    img = torch.tensor(img).to(device).unsqueeze(0)
    ex_pose = torch.tensor(ex_pose).to(device).unsqueeze(0)

    save_image(
        img,
        os.path.join(
            opt.output_dir,
            f"gt_{img_num}.png",
        ),
        nrow=5,
        normalize=True,
    )

    for seed in opt.seeds:
        frames = []
        depths = []
        output_name = f'{img_num}-{seed}.mp4'
        writer = skvideo.io.FFmpegWriter(os.path.join(opt.output_dir, output_name), outputdict={'-pix_fmt': 'yuv420p', '-crf': '21'})
        torch.manual_seed(seed)
        z = torch.randn(1, 256, device=device)
        with torch.no_grad():
            for pitch, yaw, fov in tqdm(trajectory):
                curriculum['h_mean'] = yaw
                curriculum['v_mean'] = pitch
                curriculum['fov'] = fov
                curriculum['h_stddev'] = 0
                curriculum['v_stddev'] = 0
                curriculum['batch_size'] = 1

                # frame, depth_map = generator.staged_forward(z, max_batch_size=opt.max_batch_size, depth_map=opt.depth_map, **curriculum)
                (
                    transformed_points,
                    transformed_ray_directions,
                    transformed_ray_origins,
                    pitch,
                    yaw,
                    transformed_ray_directions_expanded,
                    z_vals,
                    cam2world,
                ) = train.generate_random_ray_points(
                    device,
                    curriculum,
                    camera_extrinsics=None,
                )
                frame, _ = generator.staged_forward(
                    z,
                    transformed_ray_origins.to(device),
                    transformed_ray_directions.to(device),
                    transformed_points.to(device),
                    transformed_ray_directions_expanded.to(device),
                    pitch.to(device),
                    yaw.to(device),
                    img.to(device),
                    ex_pose.to(device),
                    z_vals.to(device),
                    **curriculum,
                )
                
                frames.append(tensor_to_PIL(frame))

            for frame in frames:
                writer.writeFrame(np.array(frame))

            writer.close()


# Same view, different seeds
for img_num in range(5):
    img, _, ex_pose = dataset[img_num]
    if is_multiview:
        img = img[0]
        ex_pose = ex_pose[0]
    # opt.input_img_num
    img = torch.tensor(img).to(device).unsqueeze(0)
    ex_pose = torch.tensor(ex_pose).to(device).unsqueeze(0)
    ex_pose = torch.cat(
        (
            ex_pose,
            torch.tensor([0, 0, 0, 1], device=device).repeat(
                ex_pose.shape[0], 1, 1
            ),
        ),
        dim=-2,
    )

    save_image(
        img,
        os.path.join(
            opt.output_dir,
            f"gt_{img_num}-seed-walk.png",
        ),
        nrow=5,
        normalize=True,
    )
    output_name = f'{img_num}-seed-walk-fixed.mp4'
    writer = skvideo.io.FFmpegWriter(os.path.join(opt.output_dir, output_name), outputdict={'-pix_fmt': 'yuv420p', '-crf': '21'})
    frames = []
    depths = []
    for seed in tqdm(range(20)):

        torch.manual_seed(seed)
        z = torch.randn(1, 256, device=device)
        pitch, yaw, fov = trajectory[10]
        curriculum['h_mean'] = yaw
        curriculum['v_mean'] = pitch
        curriculum['fov'] = fov
        curriculum['h_stddev'] = 0
        curriculum['v_stddev'] = 0
        curriculum['batch_size'] = 1

        with torch.no_grad():
            # frame, depth_map = generator.staged_forward(z, max_batch_size=opt.max_batch_size, depth_map=opt.depth_map, **curriculum)
            (
                transformed_points,
                transformed_ray_directions,
                transformed_ray_origins,
                pitch,
                yaw,
                transformed_ray_directions_expanded,
                z_vals,
                cam2world,
            ) = train.generate_random_ray_points(
                device,
                curriculum,
                camera_extrinsics=None #ex_pose,
            )
            frame, _ = generator.staged_forward(
                z,
                transformed_ray_origins.to(device),
                transformed_ray_directions.to(device),
                transformed_points.to(device),
                transformed_ray_directions_expanded.to(device),
                pitch.to(device),
                yaw.to(device),
                img.to(device),
                ex_pose.to(device),
                z_vals.to(device),
                **curriculum,
            )
            
            frames.append(tensor_to_PIL(frame))

    for frame in frames:
        writer.writeFrame(np.array(frame))

    writer.close()