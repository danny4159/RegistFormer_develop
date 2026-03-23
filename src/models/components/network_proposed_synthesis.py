import torch
import torch.nn as nn
import torch.nn.functional as F

def get_layer_by_dim(is_3d):
    dim = 3 if is_3d else 2
    Conv = getattr(nn, f'Conv{dim}d')
    Norm = getattr(nn, f'InstanceNorm{dim}d')
    return Conv, Norm, dim


class LocalStyleEncoder(nn.Module):
    """
    Local style encoder that extracts style features from reference images.
    Uses depthwise separable convolutions to limit receptive field and prevent
    structure information leakage.

    Key design choices:
    - Small receptive field to capture local texture/style
    - Depthwise separable conv to reduce parameters and prevent structure encoding
    - Low-pass tendency through moderate downsampling
    """
    def __init__(self, in_ch=1, hid=32, out_ch=64, is_3d=False):
        super().__init__()
        Conv, _, _ = get_layer_by_dim(is_3d)

        self.net = nn.Sequential(
            Conv(in_ch, hid, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            Conv(hid, hid, 3, 1, 1, groups=hid),  # depthwise
            Conv(hid, out_ch, 1, 1, 0),  # pointwise
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SharedPrivateStyleDecomposer(nn.Module):
    """
    Decomposes style features into shared and private components.

    - s_common: shared style components between two references
    - s_priv_b, s_priv_c: private style components for each reference

    The shared component captures common texture/style patterns,
    while private components capture reference-specific characteristics.
    """
    def __init__(self, ch=64, is_3d=False):
        super().__init__()
        Conv, _, _ = get_layer_by_dim(is_3d)

        # Stronger disentanglement: predict the common component from each branch first.
        self.common_from_b = Conv(ch, ch, 1, 1, 0)
        self.common_from_c = Conv(ch, ch, 1, 1, 0)

    def forward(self, s_b, s_c):
        common_b = self.common_from_b(s_b)
        common_c = self.common_from_c(s_c)
        s_common = 0.5 * (common_b + common_c)

        s_priv_b = s_b - common_b
        s_priv_c = s_c - common_c

        return s_common, common_b, common_c, s_priv_b, s_priv_c


class StyleRouter(nn.Module):
    """
    Routes shared style information to each branch with gated control.

    Key design: Conservative routing
    - Gate values start small (biased towards private style)
    - Only truly helpful shared style gets mixed in during training
    - Prevents structure information from leaking through shared pathway

    Final style for each branch:
    s_hat = s_priv + alpha * route(s_common)
    where alpha is a learned spatial gate in [0, 1]
    """
    def __init__(self, ch=64, is_3d=False):
        super().__init__()
        Conv, _, _ = get_layer_by_dim(is_3d)

        # Routing projectors for shared style
        self.route_b = Conv(ch, ch, 3, 1, 1)
        self.route_c = Conv(ch, ch, 3, 1, 1)

        # Gating networks - initialized to produce small values
        self.gate_b = Conv(ch, ch, 1, 1, 0)
        self.gate_c = Conv(ch, ch, 1, 1, 0)

        # Initialize gate biases to negative values for conservative start
        nn.init.constant_(self.gate_b.bias, -2.0)
        nn.init.constant_(self.gate_c.bias, -2.0)

    def forward(self, s_common, s_priv_b, s_priv_c):
        # Compute gating values (sigmoid ensures [0, 1])
        alpha_b = torch.sigmoid(self.gate_b(s_priv_b))
        alpha_c = torch.sigmoid(self.gate_c(s_priv_c))

        # Route shared style through projectors
        routed_b = self.route_b(s_common)
        routed_c = self.route_c(s_common)
        shared_contrib_b = alpha_b * routed_b
        shared_contrib_c = alpha_c * routed_c

        # Final style: private + gated shared
        s_hat_b = s_priv_b + shared_contrib_b
        s_hat_c = s_priv_c + shared_contrib_c

        return s_hat_b, s_hat_c, alpha_b, alpha_c, shared_contrib_b, shared_contrib_c


class ProposedSynthesisModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        try:
            self.input_nc = kwargs['input_nc']
            self.feat_ch = kwargs['feat_ch']
            self.output_nc = kwargs['output_nc']
            self.demodulate = kwargs['demodulate']
            self.use_multiple_outputs = kwargs.get('use_multiple_outputs', None)
            self.is_3d = kwargs.get('is_3d', False)
            self.use_separate_style_layers = kwargs.get('use_separate_style_layers', False)
            self.use_style_decomposition = kwargs.get('use_style_decomposition', False)
            self.style_ch = kwargs.get('style_ch', 64)

        except KeyError as e:
            raise ValueError(f"Missing required parameter: {str(e)}")

        Conv, _, _ = get_layer_by_dim(self.is_3d)

        ch = 2 if self.use_multiple_outputs else 1
        branch_output_nc = max(1, self.output_nc // 2) if self.use_multiple_outputs else self.output_nc

        if self.use_style_decomposition and self.use_multiple_outputs:
            self.local_style_encoder = LocalStyleEncoder(
                in_ch=1, hid=32, out_ch=self.style_ch, is_3d=self.is_3d
            )
            self.style_decomposer = SharedPrivateStyleDecomposer(
                ch=self.style_ch, is_3d=self.is_3d
            )
            self.style_router = StyleRouter(
                ch=self.style_ch, is_3d=self.is_3d
            )
            # Use depthwise conv to smooth style while preserving multi-channel representation
            self.style_smoother = nn.Sequential(
                Conv(self.style_ch, self.style_ch, 3, 1, 1, groups=self.style_ch),
            )

        self.aux = {}

        if self.use_multiple_outputs:
            # Shared content encoder: explicit style exchange should happen in the router,
            # not inside a shared style-conditioned trunk.
            self.conv0_shared = StyleConv(
                self.input_nc, self.feat_ch, kernel_size=3,
                activate=True, demodulate=self.demodulate,
                style_denorm=False, ch=1, use_noise=False, is_3d=self.is_3d
            )
            self.conv11_shared = StyleConv(
                self.feat_ch, self.feat_ch, kernel_size=3,
                downsample=True, activate=True, demodulate=self.demodulate,
                style_denorm=False, ch=1, use_noise=False, is_3d=self.is_3d
            )
            self.conv12_shared = StyleConv(
                self.feat_ch, self.feat_ch, kernel_size=3,
                downsample=False, activate=True, demodulate=self.demodulate,
                style_denorm=False, ch=1, use_noise=False, is_3d=self.is_3d
            )
            self.conv21_shared = StyleConv(
                self.feat_ch, self.feat_ch, kernel_size=3,
                downsample=True, activate=True, demodulate=self.demodulate,
                style_denorm=False, ch=1, use_noise=False, is_3d=self.is_3d
            )
            self.conv22_shared = StyleConv(
                self.feat_ch, self.feat_ch, kernel_size=3,
                downsample=False, activate=True, demodulate=self.demodulate,
                style_denorm=False, ch=1, use_noise=False, is_3d=self.is_3d
            )

            # Branch b StyleConvs with multi-channel style input
            style_in = self.style_ch if self.use_style_decomposition else 1
            self.conv31_b = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                      downsample=False, activate=True, demodulate=self.demodulate, ch=1, style_in_ch=style_in, is_3d=self.is_3d)
            self.conv32_b = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                      downsample=False, activate=True, demodulate=self.demodulate, ch=1, style_in_ch=style_in, is_3d=self.is_3d)
            self.conv41_b = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                      upsample=True, activate=True, demodulate=self.demodulate, ch=1, style_in_ch=style_in, is_3d=self.is_3d)
            self.conv42_b = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                      upsample=False, activate=True, demodulate=self.demodulate, ch=1, style_in_ch=style_in, is_3d=self.is_3d)
            self.conv51_b = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                      upsample=True, activate=True, demodulate=self.demodulate, ch=1, style_in_ch=style_in, is_3d=self.is_3d)
            self.conv52_b = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                      upsample=False, activate=True, demodulate=self.demodulate, ch=1, style_in_ch=style_in, is_3d=self.is_3d)
            self.conv6_b = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, style_in_ch=style_in, is_3d=self.is_3d)

            # Branch c StyleConvs with multi-channel style input
            self.conv31_c = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                      downsample=False, activate=True, demodulate=self.demodulate, ch=1, style_in_ch=style_in, is_3d=self.is_3d)
            self.conv32_c = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                      downsample=False, activate=True, demodulate=self.demodulate, ch=1, style_in_ch=style_in, is_3d=self.is_3d)
            self.conv41_c = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                      upsample=True, activate=True, demodulate=self.demodulate, ch=1, style_in_ch=style_in, is_3d=self.is_3d)
            self.conv42_c = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                      upsample=False, activate=True, demodulate=self.demodulate, ch=1, style_in_ch=style_in, is_3d=self.is_3d)
            self.conv51_c = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                      upsample=True, activate=True, demodulate=self.demodulate, ch=1, style_in_ch=style_in, is_3d=self.is_3d)
            self.conv52_c = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                      upsample=False, activate=True, demodulate=self.demodulate, ch=1, style_in_ch=style_in, is_3d=self.is_3d)
            self.conv6_c = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, style_in_ch=style_in, is_3d=self.is_3d)

            if self.use_separate_style_layers:
                self.conv7_1 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                         activate=True, demodulate=self.demodulate, ch=1, style_in_ch=style_in, is_3d=self.is_3d)
                self.conv7_2 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                         activate=True, demodulate=self.demodulate, ch=1, style_in_ch=style_in, is_3d=self.is_3d)
                self.conv8_1 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                         activate=True, demodulate=self.demodulate, ch=1, style_in_ch=style_in, is_3d=self.is_3d)
                self.conv8_2 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                         activate=True, demodulate=self.demodulate, ch=1, style_in_ch=style_in, is_3d=self.is_3d)

            self.conv_final_1 = Conv(self.feat_ch, branch_output_nc, kernel_size=3, padding=1)
            self.conv_final_2 = Conv(self.feat_ch, branch_output_nc, kernel_size=3, padding=1)
        else:
            self.conv0 = StyleConv(self.input_nc, self.feat_ch * ch, kernel_size=3,
                                   activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d)
            self.conv11 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                    downsample=True, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d)
            self.conv12 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                    downsample=False, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d)
            self.conv21 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                    downsample=True, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d)
            self.conv22 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                    downsample=False, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d)
            self.conv31 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                    downsample=False, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d)
            self.conv32 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                    downsample=False, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d)
            self.conv41 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                    upsample=True, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d)
            self.conv42 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                    upsample=False, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d)
            self.conv51 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                    upsample=True, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d)
            self.conv52 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                    upsample=False, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d)
            self.conv6 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                   activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d)
            self.conv_final = Conv(self.feat_ch * ch, self.output_nc, kernel_size=3, padding=1)

    def forward(self, merged_input, layers=[], encode_only=False):
        x = merged_input[:, :1, ...]
        ref = merged_input[:, 1:, ...]

        if self.is_3d:
            x = x.permute(0, 1, 4, 2, 3)
            ref = ref.permute(0, 1, 4, 2, 3)
            style_guidance_raw = F.interpolate(
                ref,
                scale_factor=(1.0, 1 / 16, 1 / 16),
                mode='trilinear',
                align_corners=False,
            )
        else:
            style_guidance_raw = F.interpolate(ref, scale_factor=1/16, mode='nearest')

        if self.use_style_decomposition and self.use_multiple_outputs and ref.shape[1] >= 2:
            style_1_raw = style_guidance_raw[:, :1, ...]
            style_2_raw = style_guidance_raw[:, 1:2, ...]

            s_b = self.local_style_encoder(style_1_raw)
            s_c = self.local_style_encoder(style_2_raw)

            s_common, common_b, common_c, s_priv_b, s_priv_c = self.style_decomposer(s_b, s_c)
            s_hat_b, s_hat_c, alpha_b, alpha_c, shared_contrib_b, shared_contrib_c = self.style_router(s_common, s_priv_b, s_priv_c)

            # Use multi-channel style representation directly (no 1-channel projection)
            style_1 = self.style_smoother(s_hat_b)
            style_2 = self.style_smoother(s_hat_c)

            self.aux = {
                "s_b": s_b,
                "s_c": s_c,
                "s_common": s_common,
                "common_b": common_b,
                "common_c": common_c,
                "s_priv_b": s_priv_b,
                "s_priv_c": s_priv_c,
                "s_hat_b": s_hat_b,
                "s_hat_c": s_hat_c,
                "shared_contrib_b": shared_contrib_b,
                "shared_contrib_c": shared_contrib_c,
                "alpha_b": alpha_b,
                "alpha_c": alpha_c,
            }
        else:
            style_1 = style_guidance_raw[:, :1, ...]
            style_2 = style_guidance_raw[:, 1:2, ...] if style_guidance_raw.shape[1] >= 2 else style_1
            self.aux = {}

        if self.use_multiple_outputs:
            dummy_style = style_1.new_zeros(style_1.shape)

            feat0 = self.conv0_shared(x, dummy_style)
            feat1 = self.conv11_shared(feat0, dummy_style)
            feat1 = self.conv12_shared(feat1, dummy_style)
            feat2 = self.conv21_shared(feat1, dummy_style)
            feat2 = self.conv22_shared(feat2, dummy_style)

            feat3_b = self.conv31_b(feat2, style_1)
            feat3_b = self.conv32_b(feat3_b, style_1)
            feat4_b = self.conv41_b(feat3_b + feat2, style_1)
            feat4_b = self.conv42_b(feat4_b, style_1)
            feat5_b = self.conv51_b(feat4_b + feat1, style_1)
            feat5_b = self.conv52_b(feat5_b, style_1)
            feat6_b = self.conv6_b(feat5_b + feat0, style_1)

            feat3_c = self.conv31_c(feat2, style_2)
            feat3_c = self.conv32_c(feat3_c, style_2)
            feat4_c = self.conv41_c(feat3_c + feat2, style_2)
            feat4_c = self.conv42_c(feat4_c, style_2)
            feat5_c = self.conv51_c(feat4_c + feat1, style_2)
            feat5_c = self.conv52_c(feat5_c, style_2)
            feat6_c = self.conv6_c(feat5_c + feat0, style_2)

            feat3 = torch.cat((feat3_b, feat3_c), dim=1)
            feat4 = torch.cat((feat4_b, feat4_c), dim=1)
            feat5 = torch.cat((feat5_b, feat5_c), dim=1)
            feat6 = torch.cat((feat6_b, feat6_c), dim=1)

            if self.use_separate_style_layers:
                feat7_1 = self.conv7_1(feat6_b, style_1)
                feat7_2 = self.conv7_2(feat6_c, style_2)
                feat8_1 = self.conv8_1(feat7_1, style_1)
                feat8_2 = self.conv8_2(feat7_2, style_2)
                out_1 = torch.tanh(self.conv_final_1(feat8_1))
                out_2 = torch.tanh(self.conv_final_2(feat8_2))
            else:
                out_1 = torch.tanh(self.conv_final_1(feat6_b))
                out_2 = torch.tanh(self.conv_final_2(feat6_c))

            out = torch.cat((out_1, out_2), dim=1)
        else:
            style_guidance_1 = torch.cat([style_1, style_2], dim=1) if style_guidance_raw.shape[1] >= 2 else style_1
            feat0 = self.conv0(x, style_guidance_1)
            feat1 = self.conv11(feat0, style_guidance_1)
            feat1 = self.conv12(feat1, style_guidance_1)
            feat2 = self.conv21(feat1, style_guidance_1)
            feat2 = self.conv22(feat2, style_guidance_1)
            feat3 = self.conv31(feat2, style_guidance_1)
            feat3 = self.conv32(feat3, style_guidance_1)
            feat4 = self.conv41(feat3 + feat2, style_guidance_1)
            feat4 = self.conv42(feat4, style_guidance_1)
            feat5 = self.conv51(feat4 + feat1, style_guidance_1)
            feat5 = self.conv52(feat5, style_guidance_1)
            feat6 = self.conv6(feat5 + feat0, style_guidance_1)
            out = self.conv_final(feat6)
            out = torch.tanh(out)

        if encode_only:
            if self.use_multiple_outputs:
                layers_dict = {
                    0: torch.cat((feat0, feat0), dim=1),
                    1: torch.cat((feat1, feat1), dim=1),
                    2: torch.cat((feat2, feat2), dim=1),
                    3: feat3,
                    4: feat4,
                    5: feat5,
                    6: feat6,
                }
                if self.use_separate_style_layers:
                    layers_dict[7] = torch.cat((feat7_1, feat7_2), dim=1)
                    layers_dict[8] = torch.cat((feat8_1, feat8_2), dim=1)
            else:
                layers_dict = {0: feat0, 1: feat1, 2: feat2, 3: feat3, 4: feat4, 5: feat5, 6: feat6}
            return [layers_dict[i] for i in layers]

        if self.is_3d:
            out = out.permute(0, 1, 3, 4, 2)
        return out

    def encode_content_only(self, x):
        """
        Extract content features without style modulation.
        Used for structure leakage loss to avoid overwriting self.aux.
        """
        if not self.use_multiple_outputs:
            raise NotImplementedError("encode_content_only only supported for use_multiple_outputs=True")

        if self.is_3d:
            x = x.permute(0, 1, 4, 2, 3)

        dummy_style = x.new_zeros((x.size(0), 1, *([1] * (x.ndim - 2))))

        feat0 = self.conv0_shared(x, dummy_style)
        feat1 = self.conv11_shared(feat0, dummy_style)
        feat1 = self.conv12_shared(feat1, dummy_style)
        feat2 = self.conv21_shared(feat1, dummy_style)
        feat2 = self.conv22_shared(feat2, dummy_style)

        return feat0, feat1, feat2


class StyleConv(nn.Module):
    def __init__(self,
                 input_nc,
                 feat_ch,
                 kernel_size,
                 upsample=False,
                 downsample=False,
                 activate=False,
                 blur_kernel=[1, 1.5, 1.5, 1],
                 demodulate=True,
                 style_denorm=True,
                 eps=1e-8,
                 ch=1,
                 style_in_ch=1,
                 use_noise=True,
                 is_3d=False):

        super(StyleConv, self).__init__()
        self.eps = eps
        self.input_nc = input_nc
        self.feat_ch = feat_ch
        self.demodulate = demodulate
        self.upsample = upsample
        self.downsample = downsample
        self.activate = activate
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.style_denorm = style_denorm
        self.style_in_ch = style_in_ch
        self.use_noise = use_noise
        self.is_3d = is_3d

        Conv, Norm, dim = get_layer_by_dim(is_3d)

        mode = 'trilinear' if is_3d else 'nearest'

        if self.upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, (pad0, pad1), upsample_factor=factor)
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=mode),
                Conv(input_nc, feat_ch, kernel_size=3, padding=1)
            )

        elif self.downsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(blur_kernel, (pad0, pad1))
            self.down = nn.Sequential(
                Conv(input_nc, feat_ch, stride=2, kernel_size=1, padding=0)
                )
        else:
            self.conv = Conv(input_nc, feat_ch, kernel_size=3, padding=1)

        self.normalize = Norm(feat_ch, affine=False)

        nhidden = 512

        if ch == 1:
            self.mlp_shared = nn.Sequential(
                Conv(style_in_ch, nhidden, kernel_size=3, padding=1),
                nn.ReLU())
            self.mlp_gamma = Conv(nhidden, feat_ch, kernel_size=3, padding=1)
            self.mlp_beta = Conv(nhidden, feat_ch, kernel_size=3, padding=1)
        elif ch == 2:
            self.mlp_shared = nn.Sequential(
                Conv(style_in_ch, nhidden, kernel_size=3, padding=1),
                nn.ReLU())
            self.mlp_gamma = Conv(nhidden, feat_ch//2, kernel_size=3, padding=1)
            self.mlp_beta = Conv(nhidden, feat_ch//2, kernel_size=3, padding=1)

            self.mlp_shared_2 = nn.Sequential(
                Conv(style_in_ch, nhidden, kernel_size=3, padding=1),
                nn.ReLU())
            self.mlp_gamma_2 = Conv(nhidden, feat_ch//2, kernel_size=3, padding=1)
            self.mlp_beta_2 = Conv(nhidden, feat_ch//2, kernel_size=3, padding=1)

        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.randomize_noise = True
        self.noise_strength = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x, style):
        if self.downsample:
            original_size = x.size()
            if x.size()[2:] != original_size[2:]:
                x = F.interpolate(x, size=original_size[2:], mode='bilinear', align_corners=True)
            x = self.down(x)
        elif self.upsample:
            x = self.up(x)
            original_size = x.size()
            if x.size()[2:] != original_size[2:]:
                x = F.interpolate(x, size=original_size[2:], mode='bilinear', align_corners=True)
        else:
            x = self.conv(x)

        # Normalize based on style_in_ch (multi-channel style support)
        x = self.normalize(x)

        if self.use_noise:
            if self.randomize_noise:
                noise = torch.randn_like(x) * self.noise_strength
            else:
                noise = torch.zeros_like(x) * self.noise_strength
            x = x + noise
        if self.style_denorm:
            mode = 'trilinear' if self.is_3d else 'nearest'
            style = F.interpolate(style, size=x.size()[2:], mode=mode)
            # Multi-channel style: use entire style tensor
            actv = self.mlp_shared(style)
            gamma = self.mlp_gamma(actv)
            beta = self.mlp_beta(actv)
            x = x * gamma + beta

        if self.activate:
            x = self.activation(x)
        return x


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super(Blur, self).__init__()
        kernel = _make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor**2)

        self.register_buffer('kernel', kernel)
        self.pad = pad

    def forward(self, x):
        orig_dtype = x.dtype
        x = x.float()
        out = upfirdn2d(x, self.kernel, padding=self.pad)
        return out.to(orig_dtype)


def _make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k
