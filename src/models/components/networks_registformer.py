import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
from torch.autograd import Variable
import math
import pdb
import os


def get_filter(filt_size=3):
    if filt_size == 1:
        a = np.array(
            [
                1.0,
            ]
        )
    elif filt_size == 2:
        a = np.array([1.0, 1.0])
    elif filt_size == 3:
        a = np.array([1.0, 2.0, 1.0])
    elif filt_size == 4:
        a = np.array([1.0, 3.0, 3.0, 1.0])
    elif filt_size == 5:
        a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
    elif filt_size == 6:
        a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
    elif filt_size == 7:
        a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt


class Downsample(nn.Module):
    def __init__(self, channels, pad_type="reflect", filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
        ]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer(
            "filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1))
        )

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, :: self.stride, :: self.stride]
            else:
                return self.pad(inp)[:, :, :: self.stride, :: self.stride]
        else:
            return F.conv2d(
                self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1]
            )


def get_pad_layer(pad_type):
    if pad_type in ["refl", "reflect"]:
        PadLayer = nn.ReflectionPad2d
    elif pad_type in ["repl", "replicate"]:
        PadLayer = nn.ReplicationPad2d
    elif pad_type == "zero":
        PadLayer = nn.ZeroPad2d
    else:
        print("Pad type [%s] not recognized" % pad_type)
    return PadLayer


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type="instance"):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "none":

        def norm_layer(x):
            return Identity()

    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == "linear":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(
                opt.n_epochs_decay + 1
            )
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1
        )
    elif opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, threshold=0.01, patience=5
        )
    elif opt.lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.n_epochs, eta_min=0
        )
    else:
        return NotImplementedError(
            "learning rate policy [%s] is not implemented", opt.lr_policy
        )
    return scheduler


def init_weights(net, init_type="normal", init_gain=0.02, debug=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if debug:
                print(classname)
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif (
            classname.find("BatchNorm2d") != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(
    net,
    init_type="normal",
    init_gain=0.02,
    gpu_ids=[],
    debug=False,
    initialize_weights=True,
):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        # if not amp:
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net


def define_G(
    feat_dim=14,
    ref_ch=1,
    src_ch=1,
    out_ch=1,
    nhead=2,
    mlp_ratio=2,
    pos_en_flag=False,
    k_size=28,
    attn_type="softmax",
    daca_loc=None,
    flow_type="voxelmorph",
    dam_type="synthesis_meta",
    fuse_type=None,
    init_type="normal",
    init_gain=0.02,
    gpu_ids=[],
    **kwargs,
    # input_nc,
    # output_nc,
    # ngf,
    # netG="resnet_4blocks",
    # norm="batch",
    # use_dropout=False,
    # no_antialias=False,
    # no_antialias_up=False,
    # opt=None,
    # fuse=True,
    # shared_decoder=False,
    # vit_name='Res-ViT-B_14',
    # fine_size=192,
):
    net = None
    net = RegistFormer(
        feat_dim=feat_dim,
        ref_ch=ref_ch,
        src_ch=src_ch,
        out_ch=out_ch,
        nhead=nhead,
        mlp_ratio=mlp_ratio,
        pos_en_flag=pos_en_flag,
        k_size=k_size,
        attn_type=attn_type,
        daca_loc=daca_loc,
        flow_type=flow_type,
        dam_type=dam_type,
        fuse_type=fuse_type,
        **kwargs,
    )

    return init_net(net, init_type, init_gain, gpu_ids, initialize_weights=True)


def define_D(
    input_nc,
    ndf,
    netD="basic",
    n_layers_D=3,
    norm="batch",
    init_type="normal",
    init_gain=0.02,
    no_antialias=False,
    gpu_ids=[],
    opt=None,
):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leaky RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == "basic":  # default PatchGAN classifier
        net = NLayerDiscriminator(
            input_nc,
            ndf,
            n_layers=3,
            norm_layer=norm_layer,
            no_antialias=no_antialias,
        )
    elif netD == "n_layers":  # more options
        net = NLayerDiscriminator(
            input_nc,
            ndf,
            n_layers_D,
            norm_layer=norm_layer,
            no_antialias=no_antialias,
        )
    elif netD == "pixel":  # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError(
            "Discriminator model name [%s] is not recognized" % netD
        )
    return init_net(
        net, init_type, init_gain, gpu_ids, initialize_weights=("stylegan2" not in netD)
    )


####################################################################################################


def resize_flow(flow, size_type, sizes, interp_mode="bilinear", align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == "ratio":
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == "shape":
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(
            f"Size type should be ratio or shape, but got type {size_type}."
        )

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow,
        size=(output_h, output_w),
        mode=interp_mode,
        align_corners=align_corners,
    )
    return resized_flow


def softmax_attention(q, k, v):
    # n x 1(k^2) x nhead x d x h x w
    h, w = q.shape[-2], q.shape[-1]

    q = q.permute(0, 2, 4, 5, 1, 3)  # n x nhead x h x w x 1 x d
    k = k.permute(0, 2, 4, 5, 3, 1)  # n x nhead x h x w x d x k^2
    v = v.permute(0, 2, 4, 5, 1, 3)  # n x nhead x h x w x k^2 x d

    N = q.shape[-1]  # scaled attention
    attn = torch.matmul(
        q / N**0.5, k
    )  # TODO: 이때 validation 튄다 [2, 2, 24, 24, 1, 784]
    attn = F.softmax(attn, dim=-1)
    output = torch.matmul(attn, v)

    output = output.permute(0, 4, 1, 5, 2, 3).squeeze(1)  # [4, 4, 16, 48, 48]
    attn = attn.permute(0, 4, 1, 5, 2, 3).squeeze(1)  # [4, 4, 25, 48, 48]

    return output, attn


# temporal for global attention.
def dotproduct_attention(q, k, v):
    # b x n x d x h x w
    h, w = q.shape[-2], q.shape[-1]

    q = q.flatten(-2).transpose(-2, -1)  # b x n x hw x d
    k = k.flatten(-2)  # b x n x d x hw
    v = v.flatten(-2).transpose(-2, -1)  # b x n x hw x d

    N = k.shape[-1]
    attn = None
    tmp = torch.matmul(k, v) / N
    output = torch.matmul(q, tmp)

    output = output.transpose(-2, -1)
    output = output.view(*output.shape[:-1], h, w)

    return output, attn


class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class double_conv_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class double_conv_up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv_up, self).__init__()
        self.conv = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class PosEnSine(nn.Module):
    """
    Code borrowed from DETR: models/positional_encoding.py
    output size: b*(2.num_pos_feats)*h*w
    """

    def __init__(self, num_pos_feats):
        super(PosEnSine, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.normalize = True
        self.scale = 2 * math.pi
        self.temperature = 10000

    def forward(self, x):
        b, c, h, w = x.shape
        not_mask = torch.ones(1, h, w, device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.repeat(b, 1, 1, 1)
        return pos


class MLP(nn.Module):
    """
    conv-based MLP layers.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.linear1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.linear2 = nn.Conv2d(hidden_features, out_features, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


class TransformerUnit(nn.Module):
    def __init__(
        self,
        feat_dim,
        n_head=8,
        pos_en_flag=True,
        mlp_ratio=2,
        k_size=5,
        attn_type="softmax",
        fuse_type=None,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.attn_type = attn_type
        self.fuse_type = fuse_type
        self.pos_en_flag = pos_en_flag

        self.pos_en = PosEnSine(self.feat_dim // 2)
        self.attn = MultiheadAttention(feat_dim, n_head, k_size=k_size)

        mlp_hidden_dim = int(feat_dim * mlp_ratio)
        self.mlp = MLP(in_features=feat_dim, hidden_features=mlp_hidden_dim)
        self.norm = nn.GroupNorm(1, self.feat_dim)

        if fuse_type:
            if fuse_type == "conv":
                self.fuse_conv = double_conv(in_ch=feat_dim, out_ch=feat_dim)
            elif fuse_type == "mask":
                self.fuse_conv = double_conv(in_ch=feat_dim, out_ch=feat_dim)

    def forward(self, q, k, v, flow, mask=None):
        if q.shape[-2:] != flow.shape[-2:]:
            # pdb.set_trace()
            flow = resize_flow(flow, "shape", q.shape[-2:])
        if mask != None and q.shape[-2:] != mask.shape[-2:]:
            # pdb.set_trace()
            mask = F.interpolate(mask, size=q.shape[-2:], mode="nearest")
        if self.pos_en_flag:
            q_pos_embed = self.pos_en(q)
            k_pos_embed = self.pos_en(k)
        else:
            q_pos_embed = 0
            k_pos_embed = 0

        # print(flow)
        # cross-multi-head attention
        # out, attn = checkpoint(self._attn_forward, q + q_pos_embed, k + k_pos_embed, v, flow)
        out, attn = self.attn(
            q=q + q_pos_embed,
            k=k + k_pos_embed,
            v=v,
            flow=flow,
            attn_type=self.attn_type,
        )
        # print(attn.shape)

        if self.fuse_type:
            if self.fuse_type == "conv":
                out = out + self.fuse_conv(q)
            elif self.fuse_type == "mask":
                try:
                    assert mask != None, "No mask found."
                except:
                    pdb.set_trace()
                out = (1 - mask) * out + mask * self.fuse_conv(q)

        # feed forward
        out = out + self.mlp(out)
        out = self.norm(out)

        return out


class Unet(nn.Module):
    def __init__(self, in_ch, feat_ch, out_ch):
        super().__init__()
        self.conv_in = single_conv(in_ch, feat_ch)

        self.conv1 = double_conv_down(feat_ch, feat_ch)
        self.conv2 = double_conv_down(feat_ch, feat_ch)
        self.conv3 = double_conv(feat_ch, feat_ch)
        self.conv4 = double_conv_up(feat_ch, feat_ch)
        self.conv5 = double_conv_up(feat_ch, feat_ch)
        self.conv6 = double_conv(feat_ch, out_ch)

    def forward(self, x):
        feat0 = self.conv_in(x)  # H, W
        feat1 = self.conv1(feat0)  # H/2, W/2
        feat2 = self.conv2(feat1)  # H/4, W/4
        feat3 = self.conv3(feat2)  # H/4, W/4
        feat3 = feat3 + feat2  # H/4
        feat4 = self.conv4(feat3)  # H/2, W/2
        feat4 = feat4 + feat1  # H/2, W/2
        feat5 = self.conv5(feat4)  # H
        feat5 = feat5 + feat0  # H
        feat6 = self.conv6(feat5)

        return feat0, feat1, feat2, feat3, feat4, feat6


class MultiheadAttention(nn.Module):
    def __init__(self, feat_dim, n_head, k_size=5, d_k=None, d_v=None):
        super().__init__()
        if d_k is None:
            d_k = feat_dim // n_head
        if d_v is None:
            d_v = feat_dim // n_head

        self.n_head = n_head
        self.k_size = k_size
        self.d_k = d_k
        self.d_v = d_v

        # pre-attention projection
        self.w_qs = nn.Conv2d(feat_dim, n_head * d_k, 1, bias=False)
        self.w_ks = nn.Conv2d(feat_dim, n_head * d_k, 1, bias=False)
        self.w_vs = nn.Conv2d(feat_dim, n_head * d_v, 1, bias=False)

        # after-attention combine heads
        self.fc = nn.Conv2d(n_head * d_v, feat_dim, 1, bias=False)

    def forward(self, q, k, v, flow, attn_type="softmax"):
        # input: n x c x h x w
        # flow: n x 2 x h x w
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        # Pass through the pre-attention projection:
        # n x c x h x w   ---->   n x (nhead*dk) x h x w
        q = self.w_qs(q)  # [2, 14, 24, 24]
        k = self.w_ks(k)
        v = self.w_vs(v)  # [2, 14, 24, 24]

        n, c, h, w = q.shape

        # ------ Sampling K and V features ---------
        sampling_grid = flow_to_grid(
            flow, self.k_size
        )  # return: src에서 각 local grid가 어떻게 움직여야 ref local grid가 되는지 # n x k**2, h, w, 2  # # [100, 48, 48, 2]
        # sampled feature
        # n x k^2 x c x h x w
        sample_k_feat = flow_guide_sampler(
            k, sampling_grid, k_size=self.k_size
        )  # [4, 25, 64, 48, 48] # ref에서 feature map의 local grid
        sample_v_feat = flow_guide_sampler(v, sampling_grid, k_size=self.k_size)

        # Reshape for multi-head attention.
        # q: n x 1 x nhead x dk x h x w
        # k,v: n x k^2 x nhead x dk x h x w
        q = q.view(n, 1, n_head, d_k, h, w)  # [2, 1, 2, 7, 24, 24]
        k = sample_k_feat.view(
            n, self.k_size**2, n_head, d_k, h, w
        )  # [2, 784, 2, 7, 24, 24]
        v = sample_v_feat.view(n, self.k_size**2, n_head, d_v, h, w)

        # -------------- Attention -----------------
        if attn_type == "softmax":
            # n x 1 x nhead x dk x h x w --> n x nhead x dv x h x w
            q, attn = softmax_attention(q, k, v)
        elif attn_type == "dot":
            q, attn = dotproduct_attention(q, k, v)
        else:
            raise NotImplementedError(f"Unknown attention type {attn_type}")

        # Concatenate all the heads together
        # n x (nhead*dv) x h x w
        q = q.reshape(n, -1, h, w)
        q = q.float()
        q = self.fc(q)  # n x c x h x w

        return q, attn


def flow_to_grid(flow, k_size=5):
    # flow (Tensor): Tensor with size (n, 2, h, w), normal value.
    # samples = flow + grid + shift
    # n, h, w, _ = flow.size()
    n, _, h, w = flow.size()  # [4, 2, 48, 48]
    padding = (k_size - 1) // 2

    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h), torch.arange(0, w)
    )  # [48, 48] , [48, 48]
    grid_y = grid_y[None, ...].expand(k_size**2, -1, -1).type_as(flow)  # [25, 48, 48]
    grid_x = grid_x[None, ...].expand(k_size**2, -1, -1).type_as(flow)

    shift = torch.arange(0, k_size).type_as(flow) - padding  # [5] -> -2,-1,0,1,2
    shift_y, shift_x = torch.meshgrid(shift, shift)  # [5,5]
    shift_y = shift_y.reshape(-1, 1, 1).expand(-1, h, w)  # k^2, h, w  # [25, 48, 48]
    shift_x = shift_x.reshape(-1, 1, 1).expand(-1, h, w)  # k^2, h, w

    samples_y = grid_y + shift_y  # k^2, h, w  # [25, 48, 48]
    samples_x = grid_x + shift_x  # k^2, h, w
    # src에서 local grid를 생성.
    samples_grid = torch.stack(
        (samples_x, samples_y), 3
    )  # k^2, h, w, 2 # # [25, 48, 48, 2]
    samples_grid = samples_grid[None, ...].expand(
        n, -1, -1, -1, -1
    )  # n, k^2, h, w, 2  # [4, 25, 48, 48, 2]

    flow = flow.permute(0, 2, 3, 1)[:, None, ...].expand(
        -1, k_size**2, -1, -1, -1
    )  # [4, 25, 48, 48, 2]

    # ref에서의 local grid로 옮기는 작업.
    vgrid = samples_grid + flow  ## [4, 25, 48, 48, 2]  [4, 25, 48, 48, 2]
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[..., 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[..., 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=4).view(-1, h, w, 2)
    # vgrid_scaled.requires_grad = False
    return vgrid_scaled  # n x k^2, h, w, 2  # [100, 48, 48, 2]


def flow_guide_sampler(
    feat,
    vgrid_scaled,
    k_size=5,
    interp_mode="bilinear",
    padding_mode="zeros",
    align_corners=True,
):  # feat: [4, 64, 48, 48]
    # feat (Tensor): Tensor with size (n, c, h, w).
    # vgrid (Tensor): Tensor with size (nk^2, h, w, 2)
    n, c, h, w = feat.size()
    feat = (
        feat.view(n, 1, c, h, w).expand(-1, k_size**2, -1, -1, -1).reshape(-1, c, h, w)
    )  # (nk^2, c, h, w)
    sample_feat = F.grid_sample(
        feat,
        vgrid_scaled,  # 점을 이동시키는게 아니라 grid를 이동시키는거다. 그래서 src-> ref 로 얻은 flow를 ref의 grid에 적용해주는거구나. OK!!
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    ).view(
        n, k_size**2, c, h, w
    )  # [4, 25, 64, 48, 48]
    return sample_feat


class RegistFormer(nn.Module):
    def __init__(
        self,
        feat_dim=128,
        ref_ch=1,
        src_ch=1,
        out_ch=1,
        nhead=8,
        mlp_ratio=2,
        pos_en_flag=True,
        k_size=5,
        attn_type="softmax",
        daca_loc=None,
        flow_type="voxelmorph",
        dam_type="dam",
        fuse_type=None,
        flow_model_path="pretrained/registration/Voxelmorph_Loss_MSE_MaskMSE1_0075.pt",
        flow_ft=False,
        dam_ft=False,
        dam_path="pretrained/synthesis/meta_synthesis_epoch_089.ckpt",
        dam_feat=64,
        main_ft=True,
    ):
        super().__init__()

        self.daca_loc = daca_loc
        self.flow_type = flow_type
        self.dam_type = dam_type

        current_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

        flow_model_path = os.path.join(project_root, flow_model_path)
        dam_path = os.path.join(project_root, dam_path)

        if self.flow_type == "voxelmorph" or self.flow_type == "zero":
            from src.models.components.voxelmorph import VxmDense

            device = "cuda"
            self.flow_estimator = VxmDense.load(path=flow_model_path, device=device)

            self.flow_estimator.to(device)
            self.flow_estimator.eval()
            for param in self.flow_estimator.parameters():
                param.requires_grad = flow_ft  # False
        else:
            raise ValueError(f"Unrecognized flow type: {self.flow_type}.")

        # Define DAM.
        if dam_ft:
            assert dam_path != None

        if self.dam_type == "synthesis_meta":
            from src.models.components.meta_synthesis import (
                SynthesisMetaModule,
            )

            self.DAM = SynthesisMetaModule(
                in_ch=src_ch,
                feat_ch=dam_feat,
                out_ch=src_ch,
                load_path=dam_path,
                requires_grad=dam_ft,
            )

        # Define feature extractor.
        # self.unet_q = Unet(src_ch, feat_dim, feat_dim)
        self.unet_q = Unet(src_ch * 2, feat_dim, feat_dim)
        self.unet_k = Unet(ref_ch, feat_dim, feat_dim)

        # Define GAM.
        self.trans_unit = nn.ModuleList(
            [
                TransformerUnit(
                    feat_dim,
                    nhead,
                    pos_en_flag,
                    mlp_ratio,
                    k_size,
                    attn_type,
                    fuse_type,
                ),
                TransformerUnit(
                    feat_dim,
                    nhead,
                    pos_en_flag,
                    mlp_ratio,
                    k_size,
                    attn_type,
                    fuse_type,
                ),
                TransformerUnit(
                    feat_dim,
                    nhead,
                    pos_en_flag,
                    mlp_ratio,
                    k_size,
                    attn_type,
                    fuse_type,
                ),
            ]
        )

        self.conv0 = double_conv(feat_dim, feat_dim)
        self.conv1 = double_conv_down(feat_dim, feat_dim)
        self.conv2 = double_conv_down(feat_dim, feat_dim)
        self.conv3 = double_conv(feat_dim, feat_dim)
        self.conv4 = double_conv_up(feat_dim, feat_dim)
        self.conv5 = double_conv_up(feat_dim, feat_dim)
        self.conv6 = nn.Sequential(
            single_conv(feat_dim, feat_dim), nn.Conv2d(feat_dim, out_ch, 3, 1, 1)
        )

        if not main_ft:
            self.eval()
            for key, param in self.named_parameters():
                if "flow_estimator" not in key and "DAM" not in key:
                    param.requires_grad = False
        else:
            self.train()
            for key, param in self.named_parameters():
                if "flow_estimator" not in key and "DAM" not in key:
                    param.requires_grad = True

    def forward(self, src, ref, mask=None):
        assert (
            src.shape == ref.shape
        ), "Shapes of source and reference images \
                                        mismatch."
        moved = None

        if not self.training:
            N, C, H, W = src.shape
            mod_size = 4
            H_pad = mod_size - H % mod_size if not H % mod_size == 0 else 0
            W_pad = mod_size - W % mod_size if not W % mod_size == 0 else 0
            src = F.pad(src, (0, W_pad, 0, H_pad), "replicate")
            ref = F.pad(ref, (0, W_pad, 0, H_pad), "replicate")

        if self.dam_type == "dam":
            src = self.DAM(src, ref)  # [4, 3, 256, 256]
        elif self.dam_type == "synthesis_meta":
            src_origin = src
            src = self.DAM(src)
        elif self.dam_type == "dam_misalign":
            src_origin = src
            ref_to_src = self.DAM_MR_CT(ref, src)
            src = self.DAM(src, ref)
        # elif self.dam_type == 'dam_misalign':
        #     src = self.DAM.netG_B(src, ref)
        else:
            raise ValueError(
                "Invalid dam_type provided. Expected 'dam' or 'synthesis_meta'."
            )

        #####################################################################
        def pad_tensor_to_multiple(tensor, height_multiple, width_multiple):
            _, _, h, w = tensor.shape
            h_pad = (height_multiple - h % height_multiple) % height_multiple
            w_pad = (width_multiple - w % width_multiple) % width_multiple

            # Pad the tensor
            padded_tensor = F.pad(
                tensor, (0, w_pad, 0, h_pad), mode="constant", value=-1
            )

            return padded_tensor, (h_pad, w_pad)

        def crop_tensor_to_original(tensor, padding):
            h_pad, w_pad = padding
            return tensor[:, :, : tensor.shape[2] - h_pad, : tensor.shape[3] - w_pad]

        #####################################################################

        # with torch.no_grad():
        #     flow = self.flow_estimator(src, ref).detach()
        if self.flow_type != "raft":  # voxelmorph, zero
            if self.dam_type == "synthesis_meta" or self.dam_type == "dam":
                src, moving_padding = pad_tensor_to_multiple(
                    src, height_multiple=768, width_multiple=576
                )
                ref, fixed_padding = pad_tensor_to_multiple(
                    ref, height_multiple=768, width_multiple=576
                )

                if self.flow_type == "zero":
                    moved, flow = self.flow_estimator(
                        ref, src, registration=True
                    )  # Zeroflow version
                else:
                    _, flow = self.flow_estimator(
                        src, ref, registration=True
                    )  # Original version # 첫번째 입력변수가 moving, 두번째 입력변수가 fixed

                moved, _ = self.flow_estimator(
                    ref, src, registration=True
                )  # ref -> src moved image 그냥 시각화하기 위해(save위해)
                moved = crop_tensor_to_original(moved, fixed_padding)
                src = crop_tensor_to_original(src, fixed_padding)
                ref = crop_tensor_to_original(ref, fixed_padding)
                flow = crop_tensor_to_original(flow, fixed_padding)
                if self.flow_type == "zero":
                    flow = torch.zeros_like(flow)  # Zeroflow version

            elif self.dam_type == "dam_misalign":
                src_origin, moving_padding = pad_tensor_to_multiple(
                    src_origin, height_multiple=768, width_multiple=576
                )
                ref_to_src, fixed_padding = pad_tensor_to_multiple(
                    ref_to_src, height_multiple=768, width_multiple=576
                )
                ref, fixed_padding = pad_tensor_to_multiple(
                    ref, height_multiple=768, width_multiple=576
                )

                _, flow = self.flow_estimator(src_origin, ref_to_src, registration=True)
                _, flow_mr = self.flow_estimator(
                    ref_to_src, src_origin, registration=True
                )
                moved = self.flow_estimator.transformer(ref, flow_mr)

                ref = crop_tensor_to_original(ref, fixed_padding)
                moved = crop_tensor_to_original(moved, fixed_padding)
                flow = crop_tensor_to_original(flow, fixed_padding)

            else:
                raise ValueError("Invalid dam_type")
        else:
            flow = self.flow_estimator(src, ref)  # src가 이동 ref가 고정

        src_lq_concat = torch.cat((src_origin, src), dim=1)
        # q_feat = self.unet_q(src)
        q_feat = self.unet_q(src_lq_concat)  # 발전시킨것

        k_feat = self.unet_k(ref)
        # k_feat = self.unet_k(moved) # 이건 zeroflow인데, moved를 key,value로 쓰기위해. 결과 꽤 좋더라? Ablation의 base에서는 빼야해

        outputs = []
        for i in range(3):
            # if i == 2:
            if (
                self.daca_loc == "end" and i != 2
            ):  # end일땐 끝에거만 cross-attention하도록
                continue

            if mask != None:
                mask = mask[:, 0:1, :, :]
                outputs.append(
                    self.trans_unit[i](
                        q_feat[i + 3], k_feat[i + 3], k_feat[i + 3], flow, mask
                    )
                )
            else:
                outputs.append(
                    self.trans_unit[i](
                        q_feat[i + 3], k_feat[i + 3], k_feat[i + 3], flow
                    )
                )

        if self.daca_loc == "end":
            f0 = self.conv0(outputs[0])  # H, W
            f1 = self.conv1(f0)  # H/2, W/2
            f2 = self.conv2(f1)  # H/4, W/4
            f3 = self.conv3(f2)  # H/4, W/4
            f3 = f3 + f2
            f4 = self.conv4(f3)  # H/2, W/2
            f4 = f4 + f1
            f5 = self.conv5(f4)  # H, W
            f5 = f5 + outputs[0] + f0

        else:
            f0 = self.conv0(outputs[2])  # H, W
            f1 = self.conv1(f0)  # H/2, W/2
            f1 = f1 + outputs[1]
            f2 = self.conv2(f1)  # H/4, W/4
            f2 = f2 + outputs[0]
            f3 = self.conv3(f2)  # H/4, W/4
            f3 = f3 + outputs[0] + f2
            f4 = self.conv4(f3)  # H/2, W/2
            f4 = f4 + outputs[1] + f1
            f5 = self.conv5(f4)  # H, W
            f5 = f5 + outputs[2] + f0

        out = self.conv6(f5)
        out = torch.tanh(out)  # 내가 추가한 코드

        if not self.training:
            out = out[:, :, :H, :W]

        # if moved is None:
        #     return out, src, out
        # else:
        #     return out, src, moved #src는 dam 결과
        return out


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(
        self,
        input_nc,
        ndf=64,
        n_layers=3,
        norm_layer=nn.BatchNorm2d,
        no_antialias=False,
    ):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        if no_antialias:
            sequence = [
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True),
            ]
        else:
            sequence = [
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw),
                nn.LeakyReLU(0.2, True),
                Downsample(ndf),
            ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            if no_antialias:
                sequence += [
                    nn.Conv2d(
                        ndf * nf_mult_prev,
                        ndf * nf_mult,
                        kernel_size=kw,
                        stride=2,
                        padding=padw,
                        bias=use_bias,
                    ),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                ]
            else:
                sequence += [
                    nn.Conv2d(
                        ndf * nf_mult_prev,
                        ndf * nf_mult,
                        kernel_size=kw,
                        stride=1,
                        padding=padw,
                        bias=use_bias,
                    ),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult),
                ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == "lsgan":
                loss += torch.mean((out0 - 0) ** 2) + torch.mean((out1 - 1) ** 2)
            elif self.gan_type == "nsgan":
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(
                    F.binary_cross_entropy(F.sigmoid(out0), all0)
                    + F.binary_cross_entropy(F.sigmoid(out1), all1)
                )
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == "lsgan":
                loss += torch.mean((out0 - 1) ** 2)  # LSGAN
            elif self.gan_type == "nsgan":
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
        ]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class PatchDiscriminator(NLayerDiscriminator):
    """Defines a PatchGAN discriminator"""

    def __init__(
        self,
        input_nc,
        ndf=64,
        n_layers=3,
        norm_layer=nn.BatchNorm2d,
        no_antialias=False,
    ):
        super().__init__(input_nc, ndf, 2, norm_layer, no_antialias)

    def forward(self, input):
        B, C, H, W = input.size(0), input.size(1), input.size(2), input.size(3)
        size = 16
        Y = H // size
        X = W // size
        input = input.view(B, C, Y, size, X, size)
        input = (
            input.permute(0, 2, 4, 1, 3, 5).contiguous().view(B * Y * X, C, size, size)
        )
        return super().forward(input)
