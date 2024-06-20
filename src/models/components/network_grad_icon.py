import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple

# import icon_registration.inverseConsistentNet as inverseConsistentNet
# import icon_registration.networks as networks
# import icon_registration.network_wrappers as network_wrappers


class GradICON(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        try:
            self.input_nc = kwargs['input_nc']
            self.feat_ch = kwargs['feat_ch']
            self.output_nc = kwargs['output_nc']
            self.demodulate = kwargs['demodulate']

        except KeyError as e:
            raise ValueError(f"Missing required parameter: {str(e)}")
        
        BATCH_SIZE = 8 #8
        SCALE = 2 

        input_shape = [BATCH_SIZE, 1, 32 * SCALE, 32 * SCALE, 32 * SCALE]

        phi = FunctionFromVectorField(
            tallUNet(unet=UNet2ChunkyMiddle, dimension=3)
        )
        psi = FunctionFromVectorField(tallUNet2(dimension=3))


        pretrained_lowres_net = DoubleNet(phi, psi)

        hires_net = DoubleNet(
            DownsampleNet(pretrained_lowres_net, dimension=3),
            FunctionFromVectorField(tallUNet2(dimension=3)),
        )
        hires_net.assign_identity_map(input_shape)


        self.fullres_net = GradientICON(
                                    DoubleNet(
                                    DownsampleNet(hires_net, dimension=3),
                                    FunctionFromVectorField(tallUNet2(dimension=3)),
                                    ),
                                    # inverseConsistentNet.ssd_only_interpolated,
                                    SSDOnlyInterpolated(),
                                    0.2,
                                    )
        
        BATCH_SIZE = 1
        SCALE = 4  # 1 IS QUARTER RES, 2 IS HALF RES, 4 IS FULL RES
        input_shape = [BATCH_SIZE, 1, 32 * SCALE, 32 * SCALE, 32 * SCALE]
        # input_shape = [BATCH_SIZE, 1, 40 * SCALE, 96 * SCALE, 96 * SCALE]#TODO: original
        self.fullres_net.assign_identity_map(input_shape)

      
    def forward(self, moving_img, fixed_img):
        loss, a, b, c, flips = self.fullres_net(moving_img, fixed_img)
        
        return out



####################################################################################################################################
####################################################################################################################################


class RegistrationModule(nn.Module):
    r"""Base class for icon modules that perform registration.

    A subclass of RegistrationModule should have a forward method that
    takes as input two images image_A and image_B, and returns a python function
    phi_AB that transforms a tensor of coordinates.

    RegistrationModule provides a method as_function that turns a tensor
    representing an image into a python function mapping a tensor of coordinates
    into a tensor of intensities :math:`\mathbb{R}^N \rightarrow \mathbb{R}` .
    Mathematically, this is what an image is anyway.

    After this class is constructed, but before it is used, you _must_ call
    assign_identity_map on it or on one of its parents to define the coordinate
    system associated with input images.

    The contract that a successful registration fulfils is:
    for a tensor of coordinates X, self.as_function(image_A)(phi_AB(X)) ~= self.as_function(image_B)(X)

    ie

    .. math::
        I^A \circ \Phi^{AB} \simeq I^B

    In particular, self.as_function(image_A)(phi_AB(self.identity_map)) ~= image_B
    """

    def __init__(self):
        super().__init__()
        self.downscale_factor = 1

    def as_function(self, image):
        """image is a tensor with shape self.input_shape.
        Returns a python function that maps a tensor of coordinates [batch x N_dimensions x ...]
        into a tensor of intensities.
        """

        return lambda coordinates: compute_warped_image_multiNC(
            image, coordinates, self.spacing, 1
        )

    def assign_identity_map(self, input_shape, parents_identity_map=None):
        self.input_shape = np.array(input_shape)
        self.input_shape[0] = 1
        self.spacing = 1.0 / (self.input_shape[2::] - 1)

        # if parents_identity_map is not None:
        #    self.identity_map = parents_identity_map
        # else:
        _id = identity_map_multiN(self.input_shape, self.spacing)
        self.register_buffer("identity_map", torch.from_numpy(_id), persistent=False)

        if self.downscale_factor != 1:
            child_shape = np.concatenate(
                [
                    self.input_shape[:2],
                    np.ceil(self.input_shape[2:] / self.downscale_factor).astype(int),
                ]
            )
        else:
            child_shape = self.input_shape
        for child in self.children():
            if isinstance(child, RegistrationModule):
                child.assign_identity_map(
                    child_shape,
                    # None if self.downscale_factor != 1 else self.identity_map,
                )

    def adjust_batch_size(self, size):
        shape = self.input_shape
        shape[0] = size
        self.assign_identity_map(shape)

    def forward(image_A, image_B):
        """Register a pair of images:
        return a python function phi_AB that warps a tensor of coordinates such that

        .. code-block:: python

            self.as_function(image_A)(phi_AB(self.identity_map)) ~= image_B

        .. math::
            I^A \circ \Phi^{AB} \simeq I^B

        :param image_A: the moving image
        :param image_B: the fixed image
        :return: :math:`\Phi^{AB}`
        """
        raise NotImplementedError()


class InverseConsistentNet(RegistrationModule):
    def __init__(self, network, similarity, lmbda):

        super().__init__()

        self.regis_net = network
        self.lmbda = lmbda
        self.similarity = similarity

    # def __call__(self, image_A, image_B) -> ICONLoss:
    #     return super().__call__(image_A, image_B)

    def forward(self, image_A, image_B):
        super().forward(image_A, image_B) 

        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True

        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_BA = self.regis_net(image_B, image_A)

        self.phi_AB_vectorfield = self.phi_AB(self.identity_map)
        self.phi_BA_vectorfield = self.phi_BA(self.identity_map)

        if getattr(self.similarity, "isInterpolated", False):
            # tag images during warping so that the similarity measure
            # can use information about whether a sample is interpolated
            # or extrapolated
            inbounds_tag = torch.zeros([image_A.shape[0]] + [1] + list(image_A.shape[2:]), device=image_A.device)
            if len(self.input_shape) - 2 == 3:
                inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
            elif len(self.input_shape) - 2 == 2:
                inbounds_tag[:, :, 1:-1, 1:-1] = 1.0
            else:
                inbounds_tag[:, :, 1:-1] = 1.0
        else:
            inbounds_tag = None

        self.warped_image_A = compute_warped_image_multiNC(
            torch.cat([image_A, inbounds_tag], axis=1) if inbounds_tag is not None else image_A,
            self.phi_AB_vectorfield,
            self.spacing,
            1,
        )
        self.warped_image_B = compute_warped_image_multiNC(
            torch.cat([image_B, inbounds_tag], axis=1) if inbounds_tag is not None else image_B,
            self.phi_BA_vectorfield,
            self.spacing,
            1,
        )

        similarity_loss = self.similarity(
            self.warped_image_A, image_B
        ) + self.similarity(self.warped_image_B, image_A)

        Iepsilon = (
            self.identity_map
            + torch.randn(*self.identity_map.shape).to(image_A.device)
            * 1
            / self.identity_map.shape[-1]
        )

        # inverse consistency one way

        approximate_Iepsilon1 = self.phi_AB(self.phi_BA(Iepsilon))

        approximate_Iepsilon2 = self.phi_BA(self.phi_AB(Iepsilon))

        inverse_consistency_loss = torch.mean(
            (Iepsilon - approximate_Iepsilon1) ** 2
        ) + torch.mean((Iepsilon - approximate_Iepsilon2) ** 2)

        transform_magnitude = torch.mean(
            (self.identity_map - self.phi_AB_vectorfield) ** 2
        )

        all_loss = self.lmbda * inverse_consistency_loss + similarity_loss

        return ICONLoss(
            all_loss,
            inverse_consistency_loss,
            similarity_loss,
            transform_magnitude,
            flips(self.phi_BA_vectorfield),
        )


ICONLoss = namedtuple(
    "ICONLoss",
    "all_loss inverse_consistency_loss similarity_loss transform_magnitude flips",
)


def flips(phi, in_percentage=False):
    if len(phi.size()) == 5:
        a = (phi[:, :, 1:, 1:, 1:] - phi[:, :, :-1, 1:, 1:]).detach()
        b = (phi[:, :, 1:, 1:, 1:] - phi[:, :, 1:, :-1, 1:]).detach()
        c = (phi[:, :, 1:, 1:, 1:] - phi[:, :, 1:, 1:, :-1]).detach()

        dV = torch.sum(torch.cross(a, b, 1) * c, axis=1, keepdims=True)
        if in_percentage:
            return torch.mean((dV < 0).float()) * 100.
        else:
            return torch.sum(dV < 0) / phi.shape[0]
    elif len(phi.size()) == 4:
        du = (phi[:, :, 1:, :-1] - phi[:, :, :-1, :-1]).detach()
        dv = (phi[:, :, :-1, 1:] - phi[:, :, :-1, :-1]).detach()
        dA = du[:, 0] * dv[:, 1] - du[:, 1] * dv[:, 0]
        if in_percentage:
            return torch.mean((dA < 0).float()) * 100.
        else:
            return torch.sum(dA < 0) / phi.shape[0]
    elif len(phi.size()) == 3:
        du = (phi[:, :, 1:] - phi[:, :, :-1]).detach()
        if in_percentage:
            return torch.mean((du < 0).float()) * 100.
        else:
            return torch.sum(du < 0) / phi.shape[0]
    else:
        raise ValueError()


class GradientICON(RegistrationModule):
    def __init__(self, network, similarity, lmbda):

        super().__init__()

        self.regis_net = network
        self.lmbda = lmbda
        self.similarity = similarity

    def compute_gradient_icon_loss(self, phi_AB, phi_BA):
        Iepsilon = (
            self.identity_map
            + torch.randn(*self.identity_map.shape).to(self.identity_map.device)
            * 1
            / self.identity_map.shape[-1]
        )

        # compute squared Frobenius of Jacobian of icon error

        direction_losses = []

        approximate_Iepsilon = phi_AB(phi_BA(Iepsilon))

        inverse_consistency_error = Iepsilon - approximate_Iepsilon

        delta = 0.001

        if len(self.identity_map.shape) == 4:
            dx = torch.Tensor([[[[delta]], [[0.0]]]]).to(self.identity_map.device)
            dy = torch.Tensor([[[[0.0]], [[delta]]]]).to(self.identity_map.device)
            direction_vectors = (dx, dy)

        elif len(self.identity_map.shape) == 5:
            dx = torch.Tensor([[[[[delta]]], [[[0.0]]], [[[0.0]]]]]).to(
                self.identity_map.device
            )
            dy = torch.Tensor([[[[[0.0]]], [[[delta]]], [[[0.0]]]]]).to(
                self.identity_map.device
            )
            dz = torch.Tensor([[[[0.0]]], [[[0.0]]], [[[delta]]]]).to(
                self.identity_map.device
            )
            direction_vectors = (dx, dy, dz)
        elif len(self.identity_map.shape) == 3:
            dx = torch.Tensor([[[delta]]]).to(self.identity_map.device)
            direction_vectors = (dx,)

        for d in direction_vectors:
            approximate_Iepsilon_d = phi_AB(phi_BA(Iepsilon + d))
            inverse_consistency_error_d = Iepsilon + d - approximate_Iepsilon_d
            grad_d_icon_error = (
                inverse_consistency_error - inverse_consistency_error_d
            ) / delta
            direction_losses.append(torch.mean(grad_d_icon_error**2))

        inverse_consistency_loss = sum(direction_losses)

        return inverse_consistency_loss

    def compute_similarity_measure(self, phi_AB, phi_BA, image_A, image_B):
        self.phi_AB_vectorfield = phi_AB(self.identity_map)
        self.phi_BA_vectorfield = phi_BA(self.identity_map)

        if getattr(self.similarity, "isInterpolated", False):
            # tag images during warping so that the similarity measure
            # can use information about whether a sample is interpolated
            # or extrapolated
            inbounds_tag = torch.zeros([image_A.shape[0]] + [1] + list(image_A.shape[2:]), device=image_A.device)
            if len(self.input_shape) - 2 == 3:
                inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
            elif len(self.input_shape) - 2 == 2:
                inbounds_tag[:, :, 1:-1, 1:-1] = 1.0
            else:
                inbounds_tag[:, :, 1:-1] = 1.0
        else:
            inbounds_tag = None

        self.warped_image_A = self.as_function(
            torch.cat([image_A, inbounds_tag], axis=1) if inbounds_tag is not None else image_A
        )(self.phi_AB_vectorfield)
        self.warped_image_B = self.as_function(
            torch.cat([image_B, inbounds_tag], axis=1) if inbounds_tag is not None else image_B
        )(self.phi_BA_vectorfield)
        similarity_loss = self.similarity(
            self.warped_image_A, image_B
        ) + self.similarity(self.warped_image_B, image_A)
        return similarity_loss

    def forward(self, image_A, image_B) -> ICONLoss:
        # print("Identity map shape:", self.identity_map.shape)
        # print("Image A shape:", image_A.shape)
        
        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True

        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_BA = self.regis_net(image_B, image_A)

        similarity_loss = self.compute_similarity_measure(
            self.phi_AB, self.phi_BA, image_A, image_B
        )

        inverse_consistency_loss = self.compute_gradient_icon_loss(
            self.phi_AB, self.phi_BA
        )

        all_loss = self.lmbda * inverse_consistency_loss + similarity_loss

        transform_magnitude = torch.mean(
            (self.identity_map - self.phi_AB_vectorfield) ** 2
        )
        return ICONLoss(
            all_loss,
            inverse_consistency_loss,
            similarity_loss,
            transform_magnitude,
            flips(self.phi_BA_vectorfield),
        )
    



class UNet2ChunkyMiddle(nn.Module):
    def __init__(self, num_layers, channels, dimension):
        super().__init__()
        self.dimension = dimension
        if dimension == 2:
            self.BatchNorm = nn.BatchNorm2d
            self.Conv = nn.Conv2d
            self.ConvTranspose = nn.ConvTranspose2d
            self.avg_pool = F.avg_pool2d
            self.interpolate_mode = "bilinear"
        else:
            self.BatchNorm = nn.BatchNorm3d
            self.Conv = nn.Conv3d
            self.ConvTranspose = nn.ConvTranspose3d
            self.avg_pool = F.avg_pool3d
            self.interpolate_mode = "trilinear"
        self.num_layers = num_layers
        down_channels = np.array(channels[0])
        up_channels_out = np.array(channels[1])
        up_channels_in = down_channels[1:] + np.concatenate([up_channels_out[1:], [0]])
        self.downConvs = nn.ModuleList([])
        self.upConvs = nn.ModuleList([])
        #        self.residues = nn.ModuleList([])
        self.batchNorms = nn.ModuleList(
            [
                self.BatchNorm(num_features=up_channels_out[_])
                for _ in range(self.num_layers)
            ]
        )
        for depth in range(self.num_layers):
            self.downConvs.append(
                self.Conv(
                    down_channels[depth],
                    down_channels[depth + 1],
                    kernel_size=3,
                    padding=1,
                    stride=2,
                )
            )
            self.upConvs.append(
                self.ConvTranspose(
                    up_channels_in[depth],
                    up_channels_out[depth],
                    kernel_size=4,
                    padding=1,
                    stride=2,
                )
            )
        #            self.residues.append(
        #                Residual(up_channels_out[depth])
        #            )
        self.lastConv = self.Conv(18, dimension, kernel_size=3, padding=1)
        torch.nn.init.zeros_(self.lastConv.weight)

        self.middle_dense = nn.ModuleList(
            [
                torch.nn.Linear(512 * 1 * 1 * 1, 128 * 1 * 1 * 1),
                torch.nn.Linear(128 * 1 * 1 * 1, 512 * 1 * 1 * 1),
                # torch.nn.Linear(512 * 2 * 3 * 3, 128 * 2 * 3 * 3), #TODO: original
                # torch.nn.Linear(128 * 2 * 3 * 3, 512 * 2 * 3 * 3), #TODO: original
            ]
        )

    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        skips = []
        for depth in range(self.num_layers):
            skips.append(x)
            y = self.downConvs[depth](F.leaky_relu(x))
            x = y + pad_or_crop(
                self.avg_pool(x, 2, ceil_mode=True), y.size(), self.dimension
            )
            y = F.layer_norm
        # print(x.shape)
        x = torch.reshape(x, (-1, 512 * 1 * 1 * 1))
        # x = torch.reshape(x, (-1, 512 * 2 * 3 * 3))#TODO: original
        x = self.middle_dense[1](F.leaky_relu(self.middle_dense[0](x)))
        x = torch.reshape(x, (-1, 512, 1, 1, 1))
        # x = torch.reshape(x, (-1, 512, 2, 3, 3))#TODO: original
        for depth in reversed(range(self.num_layers)):
            y = self.upConvs[depth](F.leaky_relu(x))
            x = y + F.interpolate(
                pad_or_crop(x, y.size(), self.dimension),
                scale_factor=2,
                mode=self.interpolate_mode,
                align_corners=False,
            )
            # x = self.residues[depth](x)
            x = self.batchNorms[depth](x)

            x = x[:, :, : skips[depth].size()[2], : skips[depth].size()[3]]
            x = torch.cat([x, skips[depth]], 1)
        x = self.lastConv(x)
        return x / 10














#######################################################################################################################################
# def
#######################################################################################################################################

def compute_warped_image_multiNC(
    I0, phi, spacing, spline_order, zero_boundary=False, use_01_input=True
):
    """Warps image.
    :param I0: image to warp, image size BxCxXxYxZ
    :param phi: map for the warping, size BxdimxXxYxZ
    :param spacing: image spacing [dx,dy,dz]
    :return: returns the warped image of size BxCxXxYxZ
    """

    dim = I0.dim() - 2
    if dim == 1:
        return _compute_warped_image_multiNC_1d(
            I0, phi, spacing, spline_order, zero_boundary, use_01_input=use_01_input
        )
    elif dim == 2:
        return _compute_warped_image_multiNC_2d(
            I0, phi, spacing, spline_order, zero_boundary, use_01_input=use_01_input
        )
    elif dim == 3:
        return _compute_warped_image_multiNC_3d(
            I0, phi, spacing, spline_order, zero_boundary, use_01_input=use_01_input
        )
    else:
        raise ValueError("Images can only be warped in dimensions 1 to 3")



def _compute_warped_image_multiNC_1d(
    I0, phi, spacing, spline_order, zero_boundary=False, use_01_input=True
):

    if spline_order not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        raise ValueError("Currently only orders 0 to 9 are supported")

    if spline_order == 0:
        stn = STN_ND_BCXYZ(
            spacing, zero_boundary, use_bilinear=False, use_01_input=use_01_input
        )
    elif spline_order == 1:
        stn = STN_ND_BCXYZ(
            spacing, zero_boundary, use_bilinear=True, use_01_input=use_01_input
        )
    else:
        stn = SplineInterpolation_ND_BCXYZ(spacing, spline_order)

    I1_warped = stn(I0, phi)

    return I1_warped



def _compute_warped_image_multiNC_2d(
    I0, phi, spacing, spline_order, zero_boundary=False, use_01_input=True
):

    if spline_order not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        raise ValueError("Currently only orders 0 to 9 are supported")

    if spline_order == 0:
        stn = STN_ND_BCXYZ(
            spacing, zero_boundary, use_bilinear=False, use_01_input=use_01_input
        )
    elif spline_order == 1:
        stn = STN_ND_BCXYZ(
            spacing, zero_boundary, use_bilinear=True, use_01_input=use_01_input
        )
    else:
        stn = SplineInterpolation_ND_BCXYZ(spacing, spline_order)

    I1_warped = stn(I0, phi)

    return I1_warped


def _compute_warped_image_multiNC_3d(
    I0, phi, spacing, spline_order, zero_boundary=False, use_01_input=True
):

    if spline_order not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        raise ValueError("Currently only orders 0 to 9 are supported")

    if spline_order == 0:
        # return get_warped_label_map(I0,phi,spacing)
        stn = STN_ND_BCXYZ(
            spacing, zero_boundary, use_bilinear=False, use_01_input=use_01_input
        )
    elif spline_order == 1:
        stn = STN_ND_BCXYZ(
            spacing, zero_boundary, use_bilinear=True, use_01_input=use_01_input
        )
    else:
        stn = SplineInterpolation_ND_BCXYZ(spacing, spline_order)

    I1_warped = stn(I0, phi)

    return I1_warped



class STN_ND_BCXYZ:
    """
    Spatial transform code for nD spatial transoforms. Uses the BCXYZ image format.
    """

    def __init__(
        self,
        spacing,
        zero_boundary=False,
        use_bilinear=True,
        use_01_input=True,
        use_compile_version=False,
    ):
        self.spacing = spacing
        """spatial dimension"""
        if use_compile_version:
            if use_bilinear:
                self.f = STNFunction_ND_BCXYZ_Compile(self.spacing, zero_boundary)
            else:
                self.f = partial(get_nn_interpolation, spacing=self.spacing)
        else:
            self.f = STNFunction_ND_BCXYZ(
                self.spacing,
                zero_boundary=zero_boundary,
                using_bilinear=use_bilinear,
                using_01_input=use_01_input,
            )

        """spatial transform function"""

    def __call__(self, input1, input2):
        """
        Simply returns the transformed input
        :param input1: image in BCXYZ format
        :param input2: map in BdimXYZ format
        :return: returns the transformed image
        """
        return self.f(input1, input2)



class STNFunction_ND_BCXYZ:
    """
    Spatial transform function for 1D, 2D, and 3D. In BCXYZ format (this IS the format used in the current toolbox).
    """

    def __init__(
        self, spacing, zero_boundary=False, using_bilinear=True, using_01_input=True
    ):
        """
        Constructor
        :param ndim: (int) spatial transformation of the transform
        """
        self.spacing = spacing
        self.ndim = len(spacing)
        # zero_boundary = False
        self.zero_boundary = "zeros" if zero_boundary else "border"
        self.mode = "bilinear" if using_bilinear else "nearest"
        self.using_01_input = using_01_input

    def forward_stn(self, input1, input2, ndim):
        if ndim == 1:
            # use 2D interpolation to mimick 1D interpolation
            # now test this for 1D
            phi_rs = input2.reshape(list(input2.size()) + [1])
            input1_rs = input1.reshape(list(input1.size()) + [1])

            phi_rs_size = list(phi_rs.size())
            phi_rs_size[1] = 2

            phi_rs_ordered = torch.zeros(
                phi_rs_size, dtype=phi_rs.dtype, device=phi_rs.device
            )
            # keep dimension 1 at zero
            phi_rs_ordered[:, 1, ...] = phi_rs[:, 0, ...]

            output_rs = torch.nn.functional.grid_sample(
                input1_rs,
                phi_rs_ordered.permute([0, 2, 3, 1]),
                mode=self.mode,
                padding_mode=self.zero_boundary,
                align_corners=True,
            )
            output = output_rs[:, :, :, 0]

        if ndim == 2:
            # todo double check, it seems no transpose is need for 2d, already in height width design
            input2_ordered = torch.zeros_like(input2)
            input2_ordered[:, 0, ...] = input2[:, 1, ...]
            input2_ordered[:, 1, ...] = input2[:, 0, ...]

            if input2_ordered.shape[0] == 1 and input1.shape[0] != 1:
                input2_ordered = input2_ordered.expand(input1.shape[0], -1, -1, -1)
            output = torch.nn.functional.grid_sample(
                input1,
                input2_ordered.permute([0, 2, 3, 1]),
                mode=self.mode,
                padding_mode=self.zero_boundary,
                align_corners=True,
            )
        if ndim == 3:
            input2_ordered = torch.zeros_like(input2)
            input2_ordered[:, 0, ...] = input2[:, 2, ...]
            input2_ordered[:, 1, ...] = input2[:, 1, ...]
            input2_ordered[:, 2, ...] = input2[:, 0, ...]
            if input2_ordered.shape[0] == 1 and input1.shape[0] != 1:
                input2_ordered = input2_ordered.expand(input1.shape[0], -1, -1, -1, -1)
            output = torch.nn.functional.grid_sample(
                input1,
                input2_ordered.permute([0, 2, 3, 4, 1]),
                mode=self.mode,
                padding_mode=self.zero_boundary,
                align_corners=True,
            )
        return output

    def __call__(self, input1, input2):
        """
        Perform the actual spatial transform
        :param input1: image in BCXYZ format
        :param input2: spatial transform in BdimXYZ format
        :return: spatially transformed image in BCXYZ format
        """

        assert len(self.spacing) + 2 == len(input2.size())
        if self.using_01_input:
            output = self.forward_stn(
                input1, scale_map(input2, input1.shape, self.spacing), self.ndim
            )
        else:
            output = self.forward_stn(input1, input2, self.ndim)
        # print(STNVal(output, ini=-1).sum())
        return output
    


def scale_map(map, sz, spacing):
    """
    Scales the map to the [-1,1]^d format
    :param map: map in BxCxXxYxZ format
    :param sz: size of image being interpolated in XxYxZ format
    :param spacing: spacing of image in XxYxZ format
    :return: returns the scaled map
    """

    map_scaled = torch.zeros_like(map)
    ndim = len(spacing)

    # This is to compensate to get back to the [-1,1] mapping of the following form
    # id[d]*=2./(sz[d]-1)
    # id[d]-=1.

    for d in range(ndim):
        if sz[d + 2] > 1:
            map_scaled[:, d, ...] = (
                map[:, d, ...] * (2.0 / (sz[d + 2] - 1.0) / spacing[d])
                - 1.0
                # map[:, d, ...] * 2.0 - 1.0
            )
        else:
            map_scaled[:, d, ...] = map[:, d, ...]

    return map_scaled


class TwoStepRegistration(RegistrationModule):
    """Combine two RegistrationModules.

    First netPhi is called on the input images, then image_A is warped with
    the resulting field, and then netPsi is called on warped A and image_B
    in order to find a residual warping. Finally, the composition of the two
    transforms is returned.
    """

    def __init__(self, netPhi, netPsi):
        super().__init__()
        self.netPhi = netPhi
        self.netPsi = netPsi

    def forward(self, image_A, image_B):
        
        # Tag for shortcutting hack. Must be set at the beginning of 
        # forward because it is not preserved by .to(config.device)
        self.identity_map.isIdentity = True
            
        phi = self.netPhi(image_A, image_B)
        psi = self.netPsi(
            self.as_function(image_A)(phi(self.identity_map)), 
            image_B
        )
        return lambda tensor_of_coordinates: phi(psi(tensor_of_coordinates))
        

DoubleNet = TwoStepRegistration




class DownsampleRegistration(RegistrationModule):
    """
    Perform registration using the wrapped RegistrationModule `net`
    at half input resolution.
    """

    def __init__(self, net, dimension):
        super().__init__()
        self.net = net
        if dimension == 2:
            self.avg_pool = F.avg_pool2d
            self.interpolate_mode = "bilinear"
        else:
            self.avg_pool = F.avg_pool3d
            self.interpolate_mode = "trilinear"
        self.dimension = dimension
        # This member variable is read by assign_identity_map when
        # walking the network tree and assigning identity_maps
        # to know that all children of this module operate at a lower
        # resolution.
        self.downscale_factor = 2

    def forward(self, image_A, image_B):

        image_A = self.avg_pool(image_A, 2, ceil_mode=True)
        image_B = self.avg_pool(image_B, 2, ceil_mode=True)
        return self.net(image_A, image_B)


DownsampleNet = DownsampleRegistration


class FunctionFromVectorField(RegistrationModule):
    """
    Wrap an inner neural network 'net' that returns a tensor of displacements
    [B x N x H x W (x D)], into a RegistrationModule that returns a function that
    transforms a tensor of coordinates
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image_A, image_B):
        tensor_of_displacements = self.net(image_A, image_B)
        displacement_field = self.as_function(tensor_of_displacements)

        def transform(coordinates):
            if hasattr(coordinates, "isIdentity") and coordinates.shape == tensor_of_displacements.shape:
                return coordinates + tensor_of_displacements
            return coordinates + displacement_field(coordinates)

        return transform
    

class UNet(nn.Module):
    def __init__(self, num_layers, channels, dimension):
        super().__init__()

        if dimension == 2:
            self.BatchNorm = nn.BatchNorm2d
            self.Conv = nn.Conv2d
            self.ConvTranspose = nn.ConvTranspose2d
        else:
            self.BatchNorm = nn.BatchNorm3d
            self.Conv = nn.Conv3d
            self.ConvTranspose = nn.ConvTranspose3d
        self.num_layers = num_layers
        down_channels = np.array(channels[0])
        up_channels_out = np.array(channels[1])
        up_channels_in = down_channels[1:] + np.concatenate([up_channels_out[1:], [0]])
        self.downConvs = nn.ModuleList([])
        self.upConvs = nn.ModuleList([])
        # self.residues = nn.ModuleList([])
        self.batchNorms = nn.ModuleList(
            [
                self.BatchNorm(num_features=up_channels_out[_])
                for _ in range(self.num_layers)
            ]
        )
        for depth in range(self.num_layers):
            self.downConvs.append(
                self.Conv(
                    down_channels[depth],
                    down_channels[depth + 1],
                    kernel_size=3,
                    padding=1,
                    stride=2,
                )
            )
            self.upConvs.append(
                self.ConvTranspose(
                    up_channels_in[depth],
                    up_channels_out[depth],
                    kernel_size=4,
                    padding=1,
                    stride=2,
                )
            )
            # self.residues.append(
            #    Residual(up_channels_out[depth])
            # )
        self.lastConv = self.Conv(18, dimension, kernel_size=3, padding=1)
        torch.nn.init.zeros_(self.lastConv.weight)

    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        skips = []
        for depth in range(self.num_layers):
            skips.append(x)
            x = F.relu(self.downConvs[depth](x))
        for depth in reversed(range(self.num_layers)):
            x = F.relu(self.upConvs[depth](x))
            x = self.batchNorms[depth](x)

            x = x[:, :, : skips[depth].size()[2], : skips[depth].size()[3]]
            x = torch.cat([x, skips[depth]], 1)
        x = self.lastConv(x)
        return x / 10


class UNet2(nn.Module):
    def __init__(self, num_layers, channels, dimension):
        super().__init__()
        self.dimension = dimension
        if dimension == 2:
            self.BatchNorm = nn.BatchNorm2d
            self.Conv = nn.Conv2d
            self.ConvTranspose = nn.ConvTranspose2d
            self.avg_pool = F.avg_pool2d
            self.interpolate_mode = "bilinear"
        else:
            self.BatchNorm = nn.BatchNorm3d
            self.Conv = nn.Conv3d
            self.ConvTranspose = nn.ConvTranspose3d
            self.avg_pool = F.avg_pool3d
            self.interpolate_mode = "trilinear"
        self.num_layers = num_layers
        down_channels = np.array(channels[0])
        up_channels_out = np.array(channels[1])
        up_channels_in = down_channels[1:] + np.concatenate([up_channels_out[1:], [0]])
        self.downConvs = nn.ModuleList([])
        self.upConvs = nn.ModuleList([])
        #        self.residues = nn.ModuleList([])
        self.batchNorms = nn.ModuleList(
            [
                self.BatchNorm(num_features=up_channels_out[_])
                for _ in range(self.num_layers)
            ]
        )
        for depth in range(self.num_layers):
            self.downConvs.append(
                self.Conv(
                    down_channels[depth],
                    down_channels[depth + 1],
                    kernel_size=3,
                    padding=1,
                    stride=2,
                )
            )
            self.upConvs.append(
                self.ConvTranspose(
                    up_channels_in[depth],
                    up_channels_out[depth],
                    kernel_size=4,
                    padding=1,
                    stride=2,
                )
            )
        #            self.residues.append(
        #                Residual(up_channels_out[depth])
        #            )
        self.lastConv = self.Conv(
            down_channels[0] + up_channels_out[0], dimension, kernel_size=3, padding=1
        )
        torch.nn.init.zeros_(self.lastConv.weight)
        torch.nn.init.zeros_(self.lastConv.bias)

    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        skips = []
        for depth in range(self.num_layers):
            skips.append(x)
            y = self.downConvs[depth](F.leaky_relu(x))
            x = y + pad_or_crop(
                self.avg_pool(x, 2, ceil_mode=True), y.size(), self.dimension
            )

        for depth in reversed(range(self.num_layers)):
            y = self.upConvs[depth](F.leaky_relu(x))
            x = y + F.interpolate(
                pad_or_crop(x, y.size(), self.dimension),
                scale_factor=2,
                mode=self.interpolate_mode,
                align_corners=False,
            )
            # x = self.residues[depth](x)
            x = self.batchNorms[depth](x)
            if self.dimension == 2:
                x = x[:, :, : skips[depth].size()[2], : skips[depth].size()[3]]
            else:
                x = x[
                    :,
                    :,
                    : skips[depth].size()[2],
                    : skips[depth].size()[3],
                    : skips[depth].size()[4],
                ]
            x = torch.cat([x, skips[depth]], 1)
        x = self.lastConv(x)
        return x / 10



def pad_or_crop(x, shape, dimension):
    y = x[:, : shape[1]]
    if x.size()[1] < shape[1]:
        if dimension == 3:
            y = F.pad(y, (0, 0, 0, 0, 0, 0, shape[1] - x.size()[1], 0))
        else:
            y = F.pad(y, (0, 0, 0, 0, shape[1] - x.size()[1], 0))
    assert y.size()[1] == shape[1]

    return y


def tallUNet(unet=UNet, dimension=2):
    return unet(
        5,
        [[2, 16, 32, 64, 256, 512], [16, 32, 64, 128, 256]],
        dimension,
    )

def tallUNet2(dimension=2, input_channels=1):
    return UNet2(
        5,
        [[input_channels*2, 16, 32, 64, 256, 512], [16, 32, 64, 128, 256]],
        dimension,
    )

class SimilarityBase:
    def __init__(self, isInterpolated=False):
        self.isInterpolated = isInterpolated

class SSDOnlyInterpolated(SimilarityBase):
    def __init__(self):
        super().__init__(isInterpolated=True)

    def __call__(self, image_A, image_B):
        if len(image_A.shape) - 2 == 3:
            dimensions_to_sum_over = [2, 3, 4]
        elif len(image_A.shape) - 2 == 2:
            dimensions_to_sum_over = [2, 3]
        elif len(image_A.shape) - 2 == 1:
            dimensions_to_sum_over = [2]

        inbounds_mask = image_A[:, -1:]
        image_A = image_A[:, :-1]
        assert image_A.shape == image_B.shape, "The shape of image_A and image_B sould be the same."

        inbounds_squared_distance = inbounds_mask * (image_A - image_B) ** 2
        sum_squared_distance = torch.sum(inbounds_squared_distance, dimensions_to_sum_over)
        divisor = torch.sum(inbounds_mask, dimensions_to_sum_over)
        ssds = sum_squared_distance / divisor
        return torch.mean(ssds)



def identity_map_multiN(sz, spacing, dtype="float32"):
    """
    Create an identity map
    :param sz: size of an image in BxCxXxYxZ format
    :param spacing: list with spacing information [sx,sy,sz]
    :param dtype: numpy data-type ('float32', 'float64', ...)
    :return: returns the identity map
    """
    dim = len(sz) - 2
    nrOfI = int(sz[0])

    if dim == 1:
        id = np.zeros([nrOfI, 1, sz[2]], dtype=dtype)
    elif dim == 2:
        id = np.zeros([nrOfI, 2, sz[2], sz[3]], dtype=dtype)
    elif dim == 3:
        id = np.zeros([nrOfI, 3, sz[2], sz[3], sz[4]], dtype=dtype)
    else:
        raise ValueError(
            "Only dimensions 1-3 are currently supported for the identity map"
        )

    for n in range(nrOfI):
        id[n, ...] = identity_map(sz[2::], spacing, dtype=dtype)

    return id


def identity_map(sz, spacing, dtype="float32"):
    """
    Returns an identity map.
    :param sz: just the spatial dimensions, i.e., XxYxZ
    :param spacing: list with spacing information [sx,sy,sz]
    :param dtype: numpy data-type ('float32', 'float64', ...)
    :return: returns the identity map of dimension dimxXxYxZ
    """
    dim = len(sz)
    if dim == 1:
        id = np.mgrid[0 : sz[0]]
    elif dim == 2:
        id = np.mgrid[0 : sz[0], 0 : sz[1]]
    elif dim == 3:
        id = np.mgrid[0 : sz[0], 0 : sz[1], 0 : sz[2]]
    else:
        raise ValueError(
            "Only dimensions 1-3 are currently supported for the identity map"
        )

    # now get it into range [0,(sz-1)*spacing]^d
    id = np.array(id.astype(dtype))
    if dim == 1:
        id = id.reshape(1, sz[0])  # add a dummy first index

    for d in range(dim):
        id[d] *= spacing[d]

        # id[d]*=2./(sz[d]-1)
        # id[d]-=1.

    # and now store it in a dim+1 array
    if dim == 1:
        idnp = np.zeros([1, sz[0]], dtype=dtype)
        idnp[0, :] = id[0]
    elif dim == 2:
        idnp = np.zeros([2, sz[0], sz[1]], dtype=dtype)
        idnp[0, :, :] = id[0]
        idnp[1, :, :] = id[1]
    elif dim == 3:
        idnp = np.zeros([3, sz[0], sz[1], sz[2]], dtype=dtype)
        idnp[0, :, :, :] = id[0]
        idnp[1, :, :, :] = id[1]
        idnp[2, :, :, :] = id[2]
    else:
        raise ValueError(
            "Only dimensions 1-3 are currently supported for the identity map"
        )

    return idnp
