"""Train pi-GAN. Supports distributed training."""

import argparse
import os
import numpy as np
import math

from collections import deque

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

from generators import generators
from discriminators import discriminators
from siren import siren
import fid_evaluation

import datasets
import curriculums
from tqdm import tqdm
from datetime import datetime
import copy

from torch_ema import ExponentialMovingAverage

from generators.volumetric_rendering import (
    get_initial_rays_trig,
    transform_sampled_points,
)

from util.camera_pose_visualizer import CameraPoseVisualizer


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def load_images(images, curriculum, device):
    return_images = []
    head = 0
    for stage in curriculum["stages"]:
        stage_images = images[head : head + stage["batch_size"]]
        stage_images = F.interpolate(
            stage_images, size=stage["img_size"], mode="bilinear", align_corners=True
        )
        return_images.append(stage_images)
        head += stage["batch_size"]
    return return_images


def z_sampler(shape, device, dist):
    if dist == "gaussian":
        z = torch.randn(shape, device=device)
    elif dist == "uniform":
        z = torch.rand(shape, device=device) * 2 - 1
    return z


def train(rank, world_size, opt):
    torch.manual_seed(0)

    setup(rank, world_size, opt.port)
    device = torch.device(rank)

    curriculum = getattr(curriculums, opt.curriculum)
    is_multiview = opt.curriculum == "Car_SRN"
    metadata = curriculums.extract_metadata(curriculum, 0)

    fixed_z = z_sampler(
        (metadata["batch_size"], 256), device="cpu", dist=metadata["z_dist"]
    )

    SIREN = getattr(siren, metadata["model"])

    CHANNELS = 3

    scaler = torch.cuda.amp.GradScaler()

    if opt.load_dir != "":
        generator = torch.load(
            os.path.join(opt.load_dir, "generator.pth"), map_location=device
        )
        discriminator = torch.load(
            os.path.join(opt.load_dir, "discriminator.pth"), map_location=device
        )
        ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
        ema2 = ExponentialMovingAverage(generator.parameters(), decay=0.9999)
        ema.load_state_dict(torch.load(os.path.join(opt.load_dir, "ema.pth")))
        ema2.load_state_dict(torch.load(os.path.join(opt.load_dir, "ema2.pth")))
    else:
        generator = getattr(generators, metadata["generator"])(
            SIREN, metadata["latent_dim"]
        ).to(device)
        discriminator = getattr(discriminators, metadata["discriminator"])().to(device)
        ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
        ema2 = ExponentialMovingAverage(generator.parameters(), decay=0.9999)

    generator_ddp = DDP(generator, device_ids=[rank], find_unused_parameters=True)
    discriminator_ddp = DDP(
        discriminator,
        device_ids=[rank],
        find_unused_parameters=True,
        broadcast_buffers=False,
    )
    generator = generator_ddp.module
    discriminator = discriminator_ddp.module

    if metadata.get("unique_lr", False):
        mapping_network_param_names = [
            name
            for name, _ in generator_ddp.module.siren.mapping_network.named_parameters()
        ]
        mapping_network_parameters = [
            p
            for n, p in generator_ddp.named_parameters()
            if n in mapping_network_param_names
        ]
        generator_parameters = [
            p
            for n, p in generator_ddp.named_parameters()
            if n not in mapping_network_param_names
        ]
        optimizer_G = torch.optim.Adam(
            [
                {"params": generator_parameters, "name": "generator"},
                {
                    "params": mapping_network_parameters,
                    "name": "mapping_network",
                    "lr": metadata["gen_lr"] * 5e-2,
                },
            ],
            lr=metadata["gen_lr"],
            betas=metadata["betas"],
            weight_decay=metadata["weight_decay"],
        )
    else:
        optimizer_G = torch.optim.Adam(
            generator_ddp.parameters(),
            lr=metadata["gen_lr"],
            betas=metadata["betas"],
            weight_decay=metadata["weight_decay"],
        )

    optimizer_D = torch.optim.Adam(
        discriminator_ddp.parameters(),
        lr=metadata["disc_lr"],
        betas=metadata["betas"],
        weight_decay=metadata["weight_decay"],
    )

    if opt.load_dir != "":
        optimizer_G.load_state_dict(
            torch.load(os.path.join(opt.load_dir, "optimizer_G.pth"))
        )
        optimizer_D.load_state_dict(
            torch.load(os.path.join(opt.load_dir, "optimizer_D.pth"))
        )
        if not metadata.get("disable_scaler", False):
            scaler.load_state_dict(torch.load(os.path.join(opt.load_dir, "scaler.pth")))

    generator_losses = []
    discriminator_losses = []

    if opt.set_step != None:
        generator.step = opt.set_step
        discriminator.step = opt.set_step

    if metadata.get("disable_scaler", False):
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    generator.set_device(device)

    # ----------
    #  Training
    # ----------

    with open(os.path.join(opt.output_dir, "options.txt"), "w") as f:
        f.write(str(opt))
        f.write("\n\n")
        f.write(str(generator))
        f.write("\n\n")
        f.write(str(discriminator))
        f.write("\n\n")
        f.write(str(curriculum))

    torch.manual_seed(rank)
    dataloader = None
    total_progress_bar = tqdm(
        total=opt.n_epochs, desc="Total progress", dynamic_ncols=True
    )
    total_progress_bar.update(discriminator.epoch)
    interior_step_bar = tqdm(dynamic_ncols=True)
    for _ in range(opt.n_epochs):
        total_progress_bar.update(1)

        metadata = curriculums.extract_metadata(curriculum, discriminator.step)

        # Set learning rates
        for param_group in optimizer_G.param_groups:
            if param_group.get("name", None) == "mapping_network":
                param_group["lr"] = metadata["gen_lr"] * 5e-2
            else:
                param_group["lr"] = metadata["gen_lr"]
            param_group["betas"] = metadata["betas"]
            param_group["weight_decay"] = metadata["weight_decay"]
        for param_group in optimizer_D.param_groups:
            param_group["lr"] = metadata["disc_lr"]
            param_group["betas"] = metadata["betas"]
            param_group["weight_decay"] = metadata["weight_decay"]

        if not dataloader or dataloader.batch_size != metadata["batch_size"]:
            dataloader, CHANNELS = datasets.get_dataset_distributed(
                metadata["dataset"], world_size, rank, **metadata
            )

            step_next_upsample = curriculums.next_upsample_step(
                curriculum, discriminator.step
            )
            step_last_upsample = curriculums.last_upsample_step(
                curriculum, discriminator.step
            )

            interior_step_bar.reset(total=(step_next_upsample - step_last_upsample))
            interior_step_bar.set_description(f"Progress to next stage")
            interior_step_bar.update((discriminator.step - step_last_upsample))

        for i, (imgs, _, ex_pose) in enumerate(dataloader):
            imgs = imgs.to(device)
            ex_pose = ex_pose.to(device)

            # Real extrinsic camera pose needed for pixelnerf encoding
            ex_pose = torch.cat(
                (
                    ex_pose,
                    torch.tensor([0, 0, 0, 1], device=device).repeat(
                        *ex_pose.shape[:-2], 1, 1
                    ),
                ),
                dim=-2,
            )
            input_imgs = imgs[:, 0, ...] if is_multiview else imgs
            input_pose = ex_pose[:, 0, ...] if is_multiview else ex_pose
            generator_gt_imgs = imgs[:, 1] if is_multiview else None
            generator_gt_pose = ex_pose[:, 1] if is_multiview else None
            discriminator_pose = ex_pose[:, 2, ...] if is_multiview else None
            discriminator_imgs = imgs[:, 2, ...] if is_multiview else None

            if discriminator.step % opt.model_save_interval == 0 and rank == 0:
                now = datetime.now()
                now = now.strftime("%d--%H:%M--")
                torch.save(
                    ema.state_dict(), os.path.join(opt.output_dir, now + "ema.pth")
                )
                torch.save(
                    ema2.state_dict(), os.path.join(opt.output_dir, now + "ema2.pth")
                )
                torch.save(
                    generator_ddp.module,
                    os.path.join(opt.output_dir, now + "generator.pth"),
                )
                torch.save(
                    discriminator_ddp.module,
                    os.path.join(opt.output_dir, now + "discriminator.pth"),
                )
                torch.save(
                    optimizer_G.state_dict(),
                    os.path.join(opt.output_dir, now + "optimizer_G.pth"),
                )
                torch.save(
                    optimizer_D.state_dict(),
                    os.path.join(opt.output_dir, now + "optimizer_D.pth"),
                )
                torch.save(
                    scaler.state_dict(),
                    os.path.join(opt.output_dir, now + "scaler.pth"),
                )
            metadata = curriculums.extract_metadata(curriculum, discriminator.step)

            if dataloader.batch_size != metadata["batch_size"]:
                break

            if scaler.get_scale() < 1:
                scaler.update(1.0)

            

            generator_ddp.train()
            discriminator_ddp.train()

            alpha = min(
                1, (discriminator.step - step_last_upsample) / (metadata["fade_steps"])
            )

            if is_multiview:
                if opt.use_I3_for_discr:
                    real_imgs = discriminator_imgs.to(device, non_blocking=True)
                else:
                    real_imgs = input_imgs.to(device, non_blocking=True)
            else:
                real_imgs = input_imgs.to(device, non_blocking=True)
            # print("real imgs - ", real_imgs.shape)

            metadata["nerf_noise"] = max(0, 1.0 - discriminator.step / 5000.0)
            
            # TRAIN DISCRIMINATOR
            with torch.cuda.amp.autocast():
                # Generate images for discriminator training
                with torch.no_grad():
                    (
                        transformed_points,
                        transformed_ray_directions,
                        transformed_ray_origins,
                        pitch,
                        yaw,
                        transformed_ray_directions_expanded,
                        z_vals,
                        _,
                    ) = generate_random_ray_points(device, metadata, camera_extrinsics=discriminator_pose)
                    z = z_sampler(
                        (real_imgs.shape[0], metadata["latent_dim"]),
                        device=device,
                        dist=metadata["z_dist"],
                    )
                    split_batch_size = z.shape[0] // metadata["batch_split"]
                    gen_imgs = []
                    gen_positions = []
                    for split in range(metadata["batch_split"]):

                        def get_split(x):
                            return x[
                                split
                                * split_batch_size : (split + 1)
                                * split_batch_size
                            ]

                        gen_imgs_split, gen_positions_split = generator_ddp(
                            get_split(z),
                            get_split(transformed_ray_origins),
                            get_split(transformed_ray_directions),
                            get_split(transformed_points),
                            get_split(transformed_ray_directions_expanded),
                            get_split(pitch),
                            get_split(yaw),
                            get_split(input_imgs),
                            get_split(input_pose),
                            get_split(z_vals),
                            **metadata,
                        )

                        gen_imgs.append(gen_imgs_split)
                        gen_positions.append(gen_positions_split)

                    gen_imgs = torch.cat(gen_imgs, axis=0)
                    gen_positions = torch.cat(gen_positions, axis=0)

                real_imgs.requires_grad = True
                r_preds, _, _ = discriminator_ddp(real_imgs, alpha, **metadata)

            if metadata["r1_lambda"] > 0:
                # Gradient penalty
                grad_real = torch.autograd.grad(
                    outputs=scaler.scale(r_preds.sum()),
                    inputs=real_imgs,
                    create_graph=True,
                )
                inv_scale = 1.0 / scaler.get_scale()
                grad_real = [p * inv_scale for p in grad_real][0]
            with torch.cuda.amp.autocast():
                if metadata["r1_lambda"] > 0:
                    grad_penalty = (
                        grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                    ).mean()
                    grad_penalty = 0.5 * metadata["r1_lambda"] * grad_penalty
                else:
                    grad_penalty = 0

                g_preds, g_pred_latent, g_pred_position = discriminator_ddp(
                    gen_imgs, alpha, **metadata
                )
                if metadata["z_lambda"] > 0 or metadata["pos_lambda"] > 0:
                    latent_penalty = (
                        torch.nn.MSELoss()(g_pred_latent, z) * metadata["z_lambda"]
                    )
                    position_penalty = (
                        torch.nn.MSELoss()(g_pred_position, gen_positions)
                        * metadata["pos_lambda"]
                    )
                    identity_penalty = latent_penalty + position_penalty
                else:
                    identity_penalty = 0

                # print("Discrimantor wt = ",  min(1, (discriminator.epoch**2 + 1)/opt.n_epochs))
                #  min(1, (discriminator.epoch**2)/opt.n_epochs) * 
                d_loss = (
                    torch.nn.functional.softplus(g_preds).mean()
                    + torch.nn.functional.softplus(-r_preds).mean()
                    + grad_penalty
                    + identity_penalty
                )
                discriminator_losses.append(d_loss.item())

            optimizer_D.zero_grad()
            scaler.scale(d_loss).backward()
            scaler.unscale_(optimizer_D)
            torch.nn.utils.clip_grad_norm_(
                discriminator_ddp.parameters(), metadata["grad_clip"]
            )
            scaler.step(optimizer_D)

            # TRAIN GENERATOR
            # Sampled randomly
            # PixelNerf stuff
            (
                transformed_points,
                transformed_ray_directions,
                transformed_ray_origins,
                pitch,
                yaw,
                transformed_ray_directions_expanded,
                z_vals,
                _,
            ) = generate_random_ray_points(device, metadata, camera_extrinsics=generator_gt_pose)

            z = z_sampler(
                (input_imgs.shape[0], metadata["latent_dim"]),
                device=device,
                dist=metadata["z_dist"],
            )

            split_batch_size = z.shape[0] // metadata["batch_split"]

            for split in range(metadata["batch_split"]):
                with torch.cuda.amp.autocast():

                    def get_split(x):
                        return x[
                            split * split_batch_size : (split + 1) * split_batch_size
                        ]

                    # Edited for pixelnerf
                    # TODO correct sizes with splits?
                    gen_imgs, gen_positions = generator_ddp(
                        get_split(z),
                        get_split(transformed_ray_origins),
                        get_split(transformed_ray_directions),
                        get_split(transformed_points),
                        get_split(transformed_ray_directions_expanded),
                        get_split(pitch),
                        get_split(yaw),
                        get_split(input_imgs),
                        get_split(input_pose),
                        get_split(z_vals),
                        **metadata,
                    )
                    g_preds, g_pred_latent, g_pred_position = discriminator_ddp(
                        gen_imgs, alpha, **metadata
                    )

                    topk_percentage = (
                        max(
                            0.99 ** (discriminator.step / metadata["topk_interval"]),
                            metadata["topk_v"],
                        )
                        if "topk_interval" in metadata and "topk_v" in metadata
                        else 1
                    )
                    topk_num = math.ceil(topk_percentage * g_preds.shape[0])

                    g_preds = torch.topk(g_preds, topk_num, dim=0).values

                    if metadata["z_lambda"] > 0 or metadata["pos_lambda"] > 0:
                        latent_penalty = (
                            torch.nn.MSELoss()(g_pred_latent, subset_z)
                            * metadata["z_lambda"]
                        )
                        position_penalty = (
                            torch.nn.MSELoss()(g_pred_position, gen_positions)
                            * metadata["pos_lambda"]
                        )
                        identity_penalty = latent_penalty + position_penalty
                    else:
                        identity_penalty = 0

                    # min(1, (discriminator.epoch**3)/opt.n_epochs) * 
                    g_loss = (
                        torch.nn.functional.softplus(-g_preds).mean() + identity_penalty
                    )
                    generator_losses.append(g_loss.item())
                    if is_multiview:
                        if opt.novel_view_weighting == "equal":
                            weighting_factor = 0.5
                        elif opt.novel_view_weighting == "interpolated":
                            weighting_factor = max(0, 1 - discriminator.step / 30000.0)
                        elif opt.novel_view_weighting == "only_discr":
                            weighting_factor = 0
                        else:
                            exit(1)
                        reconstruction_loss_generator = torch.nn.MSELoss()(gen_imgs, get_split(generator_gt_imgs))
                        g_loss = weighting_factor * reconstruction_loss_generator + (1-weighting_factor) * g_loss

                scaler.scale(g_loss).backward()

            
            # scaler.unscale_(optimizer_G)
            # torch.nn.utils.clip_grad_norm_(
            #     generator_ddp.parameters(), metadata.get("grad_clip", 1.3)
            # )
            # tmp as other stuff commented out
            # d_loss = torch.zeros(1)
            # g_loss = torch.zeros(1)
            # topk_num = input_imgs.shape[0]


            # Train generator with L2 loss on GT views
            (
                transformed_points,
                transformed_ray_directions,
                transformed_ray_origins,
                pitch,
                yaw,
                transformed_ray_directions_expanded,
                z_vals,
                _,
            ) = generate_random_ray_points(device, metadata, camera_extrinsics=input_pose)

            # z = z_sampler(
            #     (input_imgs.shape[0], metadata["latent_dim"]),
            #     device=device,
            #     dist=metadata["z_dist"],
            # )

            split_batch_size = z.shape[0] // metadata["batch_split"]

            for split in range(metadata["batch_split"]):
                with torch.cuda.amp.autocast():
                    def get_split(x):
                        return x[split * split_batch_size : (split + 1) * split_batch_size]

                    # Edited for pixelnerf
                    # TODO correct sizes with splits?
                    gen_imgs, gen_positions = generator_ddp(
                        get_split(z),
                        get_split(transformed_ray_origins),
                        get_split(transformed_ray_directions),
                        get_split(transformed_points),
                        get_split(transformed_ray_directions_expanded),
                        get_split(pitch),
                        get_split(yaw),
                        get_split(input_imgs),
                        get_split(input_pose),
                        get_split(z_vals),
                        **metadata,
                    )

                    reconstruction_loss = torch.nn.MSELoss()(gen_imgs, get_split(input_imgs))

                    generator_losses.append(reconstruction_loss.item())

                scaler.scale(reconstruction_loss).backward()

            scaler.unscale_(optimizer_G)
            torch.nn.utils.clip_grad_norm_(
                generator_ddp.parameters(), metadata.get("grad_clip", 0.3)
            )

            # Scale whole loss from adversarial loss and L2 loss
            scaler.step(optimizer_G)
            scaler.update()
            optimizer_G.zero_grad()
            ema.update(generator_ddp.parameters())
            ema2.update(generator_ddp.parameters())
 
            split_batch_size = input_imgs.shape[0] // metadata["batch_split"]

            if rank == 0:
                interior_step_bar.update(1)
                if i % 10 == 0:
                    multiview_part = "" if not is_multiview else f"[L2 loss generator: {reconstruction_loss_generator.item()}] "
                    tqdm.write(
                        f"[Experiment: {opt.output_dir}] [GPU: {os.environ['CUDA_VISIBLE_DEVICES']}] [Epoch: {discriminator.epoch}/{opt.n_epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}] [L2 loss: {reconstruction_loss.item()}] {multiview_part}[Step: {discriminator.step}] [Alpha: {alpha:.2f}] [Img Size: {metadata['img_size']}] [Batch Size: {metadata['batch_size']}] [TopK: {topk_num}] [Scale: {scaler.get_scale()}]"
                    )

                if discriminator.step % opt.sample_interval == 0:
                    copied_metadata = copy.deepcopy(metadata)

                    def create_image(metadata, pathname, use_gt_views=False):
                        camera_extrinsics = input_pose if use_gt_views else None
                        generator_ddp.eval()
                        with torch.no_grad():
                            gen_imgs = []
                            for split in range(metadata["batch_split"]):
                                with torch.cuda.amp.autocast():
                                    def get_split(x):
                                        return x[split * split_batch_size : (split + 1) * split_batch_size]
                                    metadata["h_stddev"] = metadata["v_stddev"] = 0
                                    (
                                        transformed_points,
                                        transformed_ray_directions,
                                        transformed_ray_origins,
                                        pitch,
                                        yaw,
                                        transformed_ray_directions_expanded,
                                        z_vals,
                                        cam2world,
                                    ) = generate_random_ray_points(
                                        device,
                                        metadata,
                                        camera_extrinsics=camera_extrinsics,
                                    )
                                    split_gen_imgs, _ = generator_ddp.module.staged_forward(
                                        get_split(fixed_z[:metadata['batch_size'], ...].to(device)),
                                        get_split(transformed_ray_origins.to(device)),
                                        get_split(transformed_ray_directions.to(device)),
                                        get_split(transformed_points.to(device)),
                                        get_split(transformed_ray_directions_expanded.to(device)),
                                        get_split(pitch.to(device)),
                                        get_split(yaw.to(device)),
                                        get_split(input_imgs.to(device)),
                                        get_split(input_pose.to(device)),
                                        get_split(z_vals.to(device)),
                                        **metadata,
                                    )
                                    gen_imgs.append(split_gen_imgs)
                            # if use_gt_views:
                            #     print("got hereeeeeeeeeeeeeeeeeeeeeeeeeeee---------------------")
                            #     for i in range(5):
                            #         visualizer = CameraPoseVisualizer(
                            #             [-1, 1], [-1, 1], [-1, 1]
                            #         )
                            #         visualizer.extrinsic2pyramid(
                            #             cam2world[i]
                            #             .cpu()
                            #             .detach()
                            #             .numpy()
                            #             .astype(np.float32),
                            #             focal_len_scaled=1.0,
                            #         )
                            #         visualizer.show(f"{pathname}_{i}__")
                        gen_imgs = torch.cat(gen_imgs, axis=0)
                        if use_gt_views:
                            save_image(
                                input_imgs[:15],
                                os.path.join(
                                    opt.output_dir,
                                    f"{discriminator.step}_gt_{pathname}.png",
                                ),
                                nrow=5,
                                normalize=True,
                            )

                        save_image(
                            gen_imgs[:15],
                            os.path.join(
                                opt.output_dir, f"{discriminator.step}_{pathname}.png"
                            ),
                            nrow=5,
                            normalize=True,
                        )

                    create_image(copied_metadata, "gt_views", use_gt_views=True)
                    create_image(copied_metadata, "fixed")
                    copied_metadata["h_mean"] += 0.5
                    create_image(copied_metadata, "tilted")
                    copied_metadata["h_mean"] -= 0.5
                    ema.store(generator_ddp.parameters())
                    ema.copy_to(generator_ddp.parameters())
                    create_image(copied_metadata, "ema_fixed")
                    copied_metadata["h_mean"] += 0.5
                    create_image(copied_metadata, "ema_tilted")
                    copied_metadata["h_mean"] -= 0.5
                    copied_metadata["psi"] = 0.7
                    create_image(copied_metadata, "random")
                    ema.restore(generator_ddp.parameters())

                if discriminator.step % opt.sample_interval == 0:
                    torch.save(
                        ema.state_dict(), os.path.join(opt.output_dir, "ema.pth")
                    )
                    torch.save(
                        ema2.state_dict(), os.path.join(opt.output_dir, "ema2.pth")
                    )
                    torch.save(
                        generator_ddp.module,
                        os.path.join(opt.output_dir, "generator.pth"),
                    )
                    torch.save(
                        discriminator_ddp.module,
                        os.path.join(opt.output_dir, "discriminator.pth"),
                    )
                    torch.save(
                        optimizer_G.state_dict(),
                        os.path.join(opt.output_dir, "optimizer_G.pth"),
                    )
                    torch.save(
                        optimizer_D.state_dict(),
                        os.path.join(opt.output_dir, "optimizer_D.pth"),
                    )
                    torch.save(
                        scaler.state_dict(), os.path.join(opt.output_dir, "scaler.pth")
                    )
                    torch.save(
                        generator_losses,
                        os.path.join(opt.output_dir, "generator.losses"),
                    )
                    torch.save(
                        discriminator_losses,
                        os.path.join(opt.output_dir, "discriminator.losses"),
                    )

            if opt.eval_freq > 0 and (discriminator.step + 1) % opt.eval_freq == 0:
                generated_dir = os.path.join(opt.output_dir, "evaluation/generated")

                if rank == 0:
                    fid_evaluation.setup_evaluation(
                        metadata["dataset"],
                        metadata["dataset_path"],
                        generated_dir,
                        target_size=128,
                    )
                dist.barrier()
                ema.store(generator_ddp.parameters())
                ema.copy_to(generator_ddp.parameters())
                generator_ddp.eval()
                fid_evaluation.output_images(
                    generator_ddp, metadata, rank, world_size, generated_dir
                )
                ema.restore(generator_ddp.parameters())
                dist.barrier()
                if rank == 0:
                    fid = fid_evaluation.calculate_fid(
                        metadata["dataset"], generated_dir, target_size=128
                    )
                    with open(os.path.join(opt.output_dir, f"fid.txt"), "a") as f:
                        f.write(f"\n{discriminator.step}:{fid}")

                torch.cuda.empty_cache()

            discriminator.step += 1
            generator.step += 1
        discriminator.epoch += 1
        generator.epoch += 1

    cleanup()


def generate_random_ray_points(device, metadata, camera_extrinsics=None):
    with torch.no_grad():
        h_stddev = metadata["h_stddev"]
        v_stddev = metadata["v_stddev"]
        h_mean = metadata["h_mean"]
        v_mean = metadata["v_mean"]
        fov = metadata["fov"]
        img_size = metadata["img_size"]
        num_steps = metadata["num_steps"]
        batch_size = metadata["batch_size"]
        sample_dist = metadata["sample_dist"]
        lock_view_dependence = metadata.get("lock_view_dependence", False)
        ray_start = metadata["ray_start"]
        ray_end = metadata["ray_end"]

        points_cam, z_vals, rays_d_cam = get_initial_rays_trig(
            batch_size,
            num_steps,
            resolution=(img_size, img_size),
            device=device,
            fov=fov,
            ray_start=ray_start,
            ray_end=ray_end,
        )  # batch_size, pixels, num_steps, 1
        (
            transformed_points,
            z_vals,
            transformed_ray_directions,
            transformed_ray_origins,
            pitch,
            yaw,
            cam2worldmatrix,
        ) = transform_sampled_points(
            points_cam,
            z_vals,
            rays_d_cam,
            h_stddev=h_stddev,
            v_stddev=v_stddev,
            h_mean=h_mean,
            v_mean=v_mean,
            device=device,
            mode=sample_dist,
            camera_extrinsics=camera_extrinsics,
        )

        transformed_ray_directions_expanded = torch.unsqueeze(
            transformed_ray_directions, -2
        )
        transformed_ray_directions_expanded = (
            transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
        )
        transformed_ray_directions_expanded = (
            transformed_ray_directions_expanded.reshape(
                batch_size, img_size * img_size * num_steps, 3
            )
        )
        transformed_points = transformed_points.reshape(
            batch_size, img_size * img_size * num_steps, 3
        )

        if lock_view_dependence:
            transformed_ray_directions_expanded = torch.zeros_like(
                transformed_ray_directions_expanded
            )
            transformed_ray_directions_expanded[..., -1] = -1
        return (
            transformed_points,
            transformed_ray_directions,
            transformed_ray_origins,
            pitch,
            yaw,
            transformed_ray_directions_expanded,
            z_vals,
            cam2worldmatrix,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_epochs", type=int, default=3000, help="number of epochs of training"
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=200,
        help="interval between image sampling",
    )
    parser.add_argument("--output_dir", type=str, default="debug")
    parser.add_argument("--load_dir", type=str, default="")
    parser.add_argument("--curriculum", type=str, required=True)
    parser.add_argument("--eval_freq", type=int, default=0)  # 5000
    parser.add_argument("--port", type=str, default="12355")
    parser.add_argument("--set_step", type=int, default=None)
    parser.add_argument("--model_save_interval", type=int, default=2000)
    parser.add_argument("--novel_view_weighting", type=str, default="equal", choices=["equal", "interpolated", "only_discr"])
    parser.add_argument("--use_I3_for_discr", action="store_true", default=False)

    opt = parser.parse_args()
    print(opt)
    os.makedirs(opt.output_dir, exist_ok=True)
    num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    mp.spawn(train, args=(num_gpus, opt), nprocs=num_gpus, join=True)
