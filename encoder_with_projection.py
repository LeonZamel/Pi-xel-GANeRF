"""
Main model implementation
"""
import torch
import numpy as np
from spacial_encoder import SpatialEncoder
from resnetfc import ResnetFC


def make_mlp(conf, d_in, d_latent=0, allow_empty=False, **kwargs):
    mlp_type = conf.get("type", "mlp")  # mlp | resnet
    if mlp_type == "mlp":
        assert False
        net = ImplicitNet.from_conf(conf, d_in + d_latent, **kwargs)
    elif mlp_type == "resnet":
        net = ResnetFC.from_conf(conf, d_in, d_latent=d_latent, **kwargs)
    elif mlp_type == "empty" and allow_empty:
        assert False
        net = None
    else:
        raise NotImplementedError("Unsupported MLP type")
    return net


def repeat_interleave(input, repeats, dim=0):
    """
    Repeat interleave along axis 0
    torch.repeat_interleave is currently very slow
    https://github.com/pytorch/pytorch/issues/31980
    """
    output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])


def make_encoder(conf, **kwargs):
    return SpatialEncoder.from_conf(conf, **kwargs)


class PositionalEncoding(torch.nn.Module):
    """
    Implement NeRF's positional encoding
    """

    def __init__(self, num_freqs=6, d_in=3, freq_factor=np.pi, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_in = d_in
        self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)
        self.d_out = self.num_freqs * 2 * d_in
        self.include_input = include_input
        if include_input:
            self.d_out += d_in
        # f1 f1 f2 f2 ... to multiply x by
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)
        )
        # 0 pi/2 0 pi/2 ... so that
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = np.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, -1, 1))

    def forward(self, x):
        """
        Apply positional encoding (new implementation)
        :param x (batch, self.d_in)
        :return (batch, self.d_out)
        """
        embed = x.unsqueeze(1).repeat(1, self.num_freqs * 2, 1)
        embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs))
        embed = embed.view(x.shape[0], -1)
        if self.include_input:
            embed = torch.cat((x, embed), dim=-1)
        return embed

    @classmethod
    def from_conf(cls, conf, d_in=3):
        # PyHocon construction
        return cls(
            conf.get("num_freqs", 6),
            d_in,
            conf.get("freq_factor", np.pi),
            conf.get("include_input", True),
        )


"""
config when using the pixelnerf on chairs dataset:
ConfigTree(
[('use_encoder', True), ('use_global_encoder', False), ('use_xyz', True), ('canon_xyz', False), ('use_code', True),
('code', ConfigTree([('num_freqs', 6), ('freq_factor', 1.5), ('include_input', True)])),
('use_viewdirs', True), ('use_code_viewdirs', False),
('mlp_coarse', ConfigTree([('type', 'resnet'), ('n_blocks', 5), ('d_hidden', 512), ('combine_layer', 3), ('combine_type', 'average')])),
('mlp_fine', ConfigTree([('type', 'resnet'), ('n_blocks', 5), ('d_hidden', 512), ('combine_layer', 3), ('combine_type', 'average')])),
('encoder', ConfigTree([('backbone', 'resnet34'), ('pretrained', True), ('num_layers', 4)]))])
"""


def get_config():
    config_dict = {
        "use_encoder": True,
        "use_global_encoder": False,
        # "use_xyz": True,
        "use_xyz": False,
        "canon_xyz": False,
        "use_code": True,
        "code": {"num_freqs": 6, "freq_factor": 1.5, "include_input": True},
        # TODO: add back
        # "use_viewdirs": True,
        "use_viewdirs": False,
        "use_code_viewdirs": False,
        "mlp_coarse": {
            "type": "resnet",
            "n_blocks": 5,
            "d_hidden": 512,
            "combine_layer": 3,
            "combine_type": "average",
        },
        "mlp_fine": {
            "type": "resnet",
            "n_blocks": 5,
            "d_hidden": 512,
            "combine_layer": 3,
            "combine_type": "average",
        },
        "encoder": {"backbone": "resnet34", "pretrained": True, "num_layers": 4},
    }
    return config_dict


class PixelNeRFEncoder(torch.nn.Module):
    def __init__(self, stop_encoder_grad=False):
        """
        :param conf PyHocon config subtree 'model'
        """
        super().__init__()
        conf = get_config()
        self.encoder = make_encoder(conf["encoder"])
        self.use_encoder = conf.get("use_encoder", True)  # Image features?

        self.use_xyz = conf.get("use_xyz", False)

        assert self.use_encoder or self.use_xyz  # Must use some feature..

        # Whether to shift z to align in canonical frame.
        # So that all objects, regardless of camera distance to center, will
        # be centered at z=0.
        # Only makes sense in ShapeNet-type setting.
        self.normalize_z = conf.get("normalize_z", True)

        self.stop_encoder_grad = (
            stop_encoder_grad  # Stop ConvNet gradient (freeze weights)
        )
        self.use_code = conf.get("use_code", False)  # Positional encoding
        self.use_code_viewdirs = conf.get(
            "use_code_viewdirs", True
        )  # Positional encoding applies to viewdirs

        # Enable view directions
        self.use_viewdirs = conf.get("use_viewdirs", False)

        # Global image features?
        self.use_global_encoder = conf.get("use_global_encoder", False)

        d_latent = self.encoder.latent_size if self.use_encoder else 0
        d_in = 3 if self.use_xyz else 1

        if self.use_viewdirs and self.use_code_viewdirs:
            # Apply positional encoding to viewdirs
            d_in += 3
        if self.use_code and d_in > 0:
            # Positional encoding for x,y,z OR view z
            self.code = PositionalEncoding.from_conf(conf["code"], d_in=d_in)
            d_in = self.code.d_out
        if self.use_viewdirs and not self.use_code_viewdirs:
            # Don't apply positional encoding to viewdirs (concat after encoded)
            d_in += 3

        d_out = 4

        self.latent_size = self.encoder.latent_size
        # self.mlp_coarse = make_mlp(conf["mlp_coarse"], d_in, d_latent, d_out=d_out)
        # self.mlp_fine = make_mlp(
        #     conf["mlp_fine"], d_in, d_latent, d_out=d_out, allow_empty=True
        # )
        # Note: this is world -> camera, and bottom row is omitted
        self.register_buffer("poses", torch.empty(1, 3, 4), persistent=False)
        self.register_buffer("image_shape", torch.empty(2), persistent=False)

        self.d_in = d_in
        self.d_out = d_out
        self.d_latent = d_latent
        self.register_buffer("focal", torch.empty(1, 2), persistent=False)
        # Principal point
        self.register_buffer("c", torch.empty(1, 2), persistent=False)

        self.num_objs = 0
        self.num_views_per_obj = 1

    def encode(self, images, poses, focal, z_bounds=None, c=None):
        """
        :param images (NS, 3, H, W)
        NS is number of input (aka source or reference) views
        :param poses (NS, 4, 4)
        :param focal focal length () or (2) or (NS) or (NS, 2) [fx, fy]
        :param z_bounds ignored argument (used in the past)
        :param c principal point None or () or (2) or (NS) or (NS, 2) [cx, cy],
        default is center of image
        """
        # invert poses to get world -> camera
        # poses = torch.inverse(poses)

        self.num_objs = images.size(0)
        if len(images.shape) == 5:
            assert len(poses.shape) == 4
            assert poses.size(1) == images.size(
                1
            )  # Be consistent with NS = num input views
            self.num_views_per_obj = images.size(1)
            images = images.reshape(-1, *images.shape[2:])
            poses = poses.reshape(-1, 4, 4)
        else:
            self.num_views_per_obj = 1

        self.encoder(images)
        rot = poses[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
        trans = -torch.bmm(rot, poses[:, :3, 3:])  # (B, 3, 1)
        self.poses = torch.cat((rot, trans), dim=-1)  # (B, 3, 4)

        self.image_shape[0] = images.shape[-1]
        self.image_shape[1] = images.shape[-2]

        # Handle various focal length/principal point formats
        if len(focal.shape) == 0:
            # Scalar: fx = fy = value for all views
            focal = focal[None, None].repeat((1, 2))
        elif len(focal.shape) == 1:
            # Vector f: fx = fy = f_i *for view i*
            # Length should match NS (or 1 for broadcast)
            focal = focal.unsqueeze(-1).repeat((1, 2))
        else:
            focal = focal.clone()
        self.focal = focal.float()
        self.focal[..., 1] *= -1.0

        if c is None:
            # Default principal point is center of image
            c = (self.image_shape * 0.5).unsqueeze(0)
        elif len(c.shape) == 0:
            # Scalar: cx = cy = value for all views
            c = c[None, None].repeat((1, 2))
        elif len(c.shape) == 1:
            # Vector c: cx = cy = c_i *for view i*
            c = c.unsqueeze(-1).repeat((1, 2))
        self.c = c

        if self.use_global_encoder:
            self.global_encoder(images)

    def forward(self, xyz, coarse=True, viewdirs=None, far=False):
        """
        Predict (r, g, b, sigma) at world space points xyz.
        Please call encode first!
        :param xyz (SB, B, 3)
        SB is batch of objects
        B is batch of points (in rays)
        NS is number of input views
        :return
        """
        SB, B, _ = xyz.shape
        NS = self.num_views_per_obj

        # plot xyz points in 3d with matplotlib
        # import numpy as np
        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        
        # print(xyz.shape)

        # Assuming your tensor is a numpy array with shape (B, 3)
        # If you're using a different library like PyTorch or TensorFlow, convert the tensor to a numpy array
        # tensor = xyz[0, ...].cpu().detach().numpy()

        # # Extract the x, y, and z coordinates
        # x = tensor[:, 0]
        # y = tensor[:, 1]
        # z = tensor[:, 2]

        # # Create a 3D plot
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # # Plot the points
        # ax.scatter(x, z, y)

        # # Label the axes
        # ax.set_xlabel('X')
        # ax.set_ylabel('Z')
        # ax.set_zlabel('Y')

        # for angle in range(0, 90, 10):
        #     ax.view_init(elev=20, azim=angle)

        #     # Show the plot
        #     # plt.show()
        #     plt.savefig(f"points/points{angle}.png")
        
        # Transform query points into the camera spaces of the input views
        xyz = repeat_interleave(xyz, NS)  # (SB*NS, B, 3)
        xyz_rot = torch.matmul(self.poses[:, None, :3, :3], xyz.unsqueeze(-1))[..., 0]
        xyz = xyz_rot + self.poses[:, None, :3, 3]

        if self.d_in > 0:
            # * Encode the xyz coordinates
            if self.use_xyz:
                if self.normalize_z:
                    z_feature = xyz_rot.reshape(-1, 3)  # (SB*B, 3)
                else:
                    z_feature = xyz.reshape(-1, 3)  # (SB*B, 3)
            else:
                if self.normalize_z:
                    z_feature = -xyz_rot[..., 2].reshape(-1, 1)  # (SB*B, 1)
                else:
                    z_feature = -xyz[..., 2].reshape(-1, 1)  # (SB*B, 1)

            if self.use_code and not self.use_code_viewdirs:
                # Positional encoding (no viewdirs)
                z_feature = self.code(z_feature)

            if self.use_viewdirs:
                # * Encode the view directions
                assert viewdirs is not None
                # Viewdirs to input view space
                viewdirs = viewdirs.reshape(SB, B, 3, 1)
                viewdirs = repeat_interleave(viewdirs, NS)  # (SB*NS, B, 3, 1)
                viewdirs = torch.matmul(
                    self.poses[:, None, :3, :3], viewdirs
                )  # (SB*NS, B, 3, 1)
                viewdirs = viewdirs.reshape(-1, 3)  # (SB*B, 3)
                z_feature = torch.cat((z_feature, viewdirs), dim=1)  # (SB*B, 4 or 6)

            if self.use_code and self.use_code_viewdirs:
                # Positional encoding (with viewdirs)
                z_feature = self.code(z_feature)

            mlp_input = z_feature

        if self.use_encoder:
            # Grab encoder's latent code.
            uv = -xyz[:, :, :2] / xyz[:, :, 2:]  # (SB, B, 2)
            uv *= repeat_interleave(
                self.focal.unsqueeze(1), NS if self.focal.shape[0] > 1 else 1
            )
            uv += repeat_interleave(
                self.c.unsqueeze(1), NS if self.c.shape[0] > 1 else 1
            )  # (SB*NS, B, 2)

            latent = self.encoder.index(
                uv, None, self.image_shape
            )  # (SB * NS, latent, B)

            if self.stop_encoder_grad:
                latent = latent.detach()
            latent = latent.transpose(1, 2).reshape(
                -1, self.latent_size
            )  # (SB * NS * B, latent)

            if self.d_in == 0:
                # z_feature not needed
                mlp_input = latent
            else:
                mlp_input = torch.cat((latent, z_feature), dim=-1)

        return mlp_input
