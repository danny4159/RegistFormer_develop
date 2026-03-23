import torch
import torch.nn as nn
import torch.nn.functional as F

def get_layer_by_dim(is_3d):
    dim = 3 if is_3d else 2
    Conv = getattr(nn, f'Conv{dim}d')
    Norm = getattr(nn, f'InstanceNorm{dim}d')
    return Conv, Norm, dim


class HighPassFilter(nn.Module):
    """Laplacian-based high-pass filter for extracting texture/style information"""
    def __init__(self, is_3d=False):
        super().__init__()
        self.is_3d = is_3d
        # Laplacian kernel for edge/texture detection
        if is_3d:
            kernel = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                   [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                                   [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=torch.float32)
            kernel = kernel.view(1, 1, 3, 3, 3)
        else:
            kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
            kernel = kernel.view(1, 1, 3, 3)
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        B, C = x.shape[:2]
        if self.is_3d:
            x_flat = x.view(B * C, 1, *x.shape[2:])
            high_freq = F.conv3d(x_flat, self.kernel, padding=1)
            high_freq = high_freq.view(B, C, *x.shape[2:])
        else:
            x_flat = x.view(B * C, 1, *x.shape[2:])
            high_freq = F.conv2d(x_flat, self.kernel, padding=1)
            high_freq = high_freq.view(B, C, *x.shape[2:])
        return high_freq


class LocalStatistics(nn.Module):
    """Extract local mean and std as style features"""
    def __init__(self, kernel_size=5, is_3d=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.is_3d = is_3d
        self.padding = kernel_size // 2

    def forward(self, x):
        if self.is_3d:
            # Adaptive kernel size for 3D to handle small dimensions
            spatial_size = x.shape[2:]  # (D, H, W) or (H, W, D) depending on permute
            k = min(self.kernel_size, min(spatial_size))
            k = max(k, 1)  # at least 1
            if k % 2 == 0:
                k = k - 1  # make it odd
            k = max(k, 1)
            p = k // 2
            local_mean = F.avg_pool3d(x, k, stride=1, padding=p)
            local_sq_mean = F.avg_pool3d(x ** 2, k, stride=1, padding=p)
        else:
            local_mean = F.avg_pool2d(x, self.kernel_size, stride=1, padding=self.padding)
            local_sq_mean = F.avg_pool2d(x ** 2, self.kernel_size, stride=1, padding=self.padding)
        local_std = torch.sqrt(torch.clamp(local_sq_mean - local_mean ** 2, min=1e-8))
        return local_mean, local_std


class CommonPrivateStyleRouter(nn.Module):
    """
    Common-Private Local Style Fusion (CPLSF) block

    Decomposes each reference style into:
    - Common style: shared texture/style across references
    - Private style: reference-specific style

    Only the common style is exchanged between references.
    """
    def __init__(self, style_ch=64, use_freq_prior=True, use_source_gate=True, is_3d=False):
        super().__init__()
        self.style_ch = style_ch
        self.use_freq_prior = use_freq_prior
        self.use_source_gate = use_source_gate
        self.is_3d = is_3d

        Conv, Norm, _ = get_layer_by_dim(is_3d)

        # Frequency prior modules
        if use_freq_prior:
            self.high_pass = HighPassFilter(is_3d)
            self.local_stats = LocalStatistics(kernel_size=5, is_3d=is_3d)
            input_ch = 4  # original + high_pass + local_mean + local_std
        else:
            input_ch = 1

        # Style encoder (shared weights for both branches)
        self.style_encoder = nn.Sequential(
            Conv(input_ch, style_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            Conv(style_ch, style_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Common/Private decomposition heads
        self.common_head = nn.Sequential(
            Conv(style_ch, style_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.private_head = nn.Sequential(
            Conv(style_ch, style_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Conservative shared style fuser
        self.shared_fuser = nn.Sequential(
            Conv(style_ch * 2, style_ch, kernel_size=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            Conv(style_ch, style_ch, kernel_size=3, padding=1, groups=style_ch),  # depthwise
            Conv(style_ch, style_ch, kernel_size=1, padding=0),
        )

        # Gate generators (conservative, initialized small)
        if use_source_gate:
            self.gate_b = nn.Sequential(
                Conv(style_ch + 1, style_ch, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                Conv(style_ch, style_ch, kernel_size=1, padding=0),
                nn.Sigmoid()
            )
            self.gate_c = nn.Sequential(
                Conv(style_ch + 1, style_ch, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                Conv(style_ch, style_ch, kernel_size=1, padding=0),
                nn.Sigmoid()
            )
        else:
            self.gate_b = nn.Sequential(
                Conv(style_ch, style_ch, kernel_size=1, padding=0),
                nn.Sigmoid()
            )
            self.gate_c = nn.Sequential(
                Conv(style_ch, style_ch, kernel_size=1, padding=0),
                nn.Sigmoid()
            )

        # Output projection back to 1 channel
        self.out_b = Conv(style_ch, 1, kernel_size=3, padding=1)
        self.out_c = Conv(style_ch, 1, kernel_size=3, padding=1)

        self._init_gates()

    def _init_gates(self):
        """Initialize gate layers to output small values initially"""
        for gate in [self.gate_b, self.gate_c]:
            for m in gate.modules():
                if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                    nn.init.constant_(m.weight, 0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, -2.0)  # sigmoid(-2) ≈ 0.12

    def _prepare_input(self, style):
        """Prepare input with optional frequency priors"""
        if self.use_freq_prior:
            high_freq = self.high_pass(style)
            local_mean, local_std = self.local_stats(style)
            style_in = torch.cat([style, high_freq, local_mean, local_std], dim=1)
        else:
            style_in = style
        return style_in

    def forward(self, style_b, style_c, source_low=None):
        """
        Args:
            style_b: [B, 1, H, W] - style from reference B (low-res)
            style_c: [B, 1, H, W] - style from reference C (low-res)
            source_low: [B, 1, H, W] - source image (low-res), for source-conditioned gating

        Returns:
            style_b_fused: [B, 1, H, W] - fused style for branch B
            style_c_fused: [B, 1, H, W] - fused style for branch C
            decomp_dict: dict for loss computation
        """
        # Prepare inputs with frequency priors
        style_b_in = self._prepare_input(style_b)
        style_c_in = self._prepare_input(style_c)

        # Encode styles
        fb = self.style_encoder(style_b_in)
        fc = self.style_encoder(style_c_in)

        # Decompose into common and private
        common_b = self.common_head(fb)
        common_c = self.common_head(fc)
        private_b = self.private_head(fb)
        private_c = self.private_head(fc)

        # Fuse common styles conservatively
        c_shared = self.shared_fuser(torch.cat([common_b, common_c], dim=1))

        # Generate gates (source-conditioned if available)
        if self.use_source_gate and source_low is not None:
            if source_low.shape[2:] != common_b.shape[2:]:
                mode = 'trilinear' if self.is_3d else 'bilinear'
                source_low = F.interpolate(source_low, size=common_b.shape[2:], mode=mode, align_corners=False)
            gate_in_b = torch.cat([common_b, source_low], dim=1)
            gate_in_c = torch.cat([common_c, source_low], dim=1)
            alpha_b = self.gate_b(gate_in_b)
            alpha_c = self.gate_c(gate_in_c)
        else:
            alpha_b = self.gate_b(common_b)
            alpha_c = self.gate_c(common_c)

        # Combine: private + gated shared
        sb = private_b + alpha_b * c_shared
        sc = private_c + alpha_c * c_shared

        # Project back to 1 channel
        style_b_fused = self.out_b(sb)
        style_c_fused = self.out_c(sc)

        decomp_dict = {
            'common_b': common_b,
            'common_c': common_c,
            'private_b': private_b,
            'private_c': private_c,
            'c_shared': c_shared,
            'alpha_b': alpha_b,
            'alpha_c': alpha_c,
        }

        return style_b_fused, style_c_fused, decomp_dict


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
            # Common-Private Style Router options
            self.use_style_router = kwargs.get('use_style_router', False)
            self.style_router_ch = kwargs.get('style_router_ch', 64)
            self.use_freq_prior = kwargs.get('use_freq_prior', True)
            self.use_source_gate = kwargs.get('use_source_gate', True)

        except KeyError as e:
            raise ValueError(f"Missing required parameter: {str(e)}")

        Conv, _, _ = get_layer_by_dim(self.is_3d)

        ch = 2 if self.use_multiple_outputs else 1

        # Common-Private Style Router (CPLSF block)
        if self.use_style_router and self.use_multiple_outputs:
            self.style_router = CommonPrivateStyleRouter(
                style_ch=self.style_router_ch,
                use_freq_prior=self.use_freq_prior,
                use_source_gate=self.use_source_gate,
                is_3d=self.is_3d
            )

        self.guide_net = nn.Sequential(
            nn.Conv2d(self.input_nc, int(self.feat_ch / 8), kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(int(self.feat_ch / 8), int(self.feat_ch / 8), kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(int(self.feat_ch / 8), int(self.feat_ch / 8), kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 일부분에만 style_denorm -> 21, 22, 31, 32에만 적용 #TODO: feat_ch에 ref ch만큼 배수로
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
        self.conv41 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3, #feat_ch *4는 변치않게
                                upsample=True, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d)
        self.conv42 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                upsample=False, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d)
        self.conv51 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                upsample=True, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d)
        self.conv52 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                upsample=False, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d)
        self.conv6 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d) # 이걸 빠트렸었어.
                                # activate=False, demodulate=self.demodulate)

        # Separate style layers: 채널을 완전히 분리해서 각각 독립적으로 style 적용
        if self.use_separate_style_layers and self.use_multiple_outputs:
            # 각 branch는 feat_ch 채널 (전체의 절반)
            self.conv7_1 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d)
            self.conv7_2 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d)
            self.conv8_1 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d)
            self.conv8_2 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d)
            # 각각 독립적인 conv_final
            self.conv_final_1 = Conv(self.feat_ch, self.output_nc // 2, kernel_size=3, padding=1)
            self.conv_final_2 = Conv(self.feat_ch, self.output_nc // 2, kernel_size=3, padding=1)
        
        ## Added for checkerboard artifact
        # if self.use_multiple_outputs:
        #     self.conv7_1 = nn.Conv2d(self.feat_ch * ch // 2, self.feat_ch * ch // 4, kernel_size=3, padding=1)
        #     self.conv7_2 = nn.Conv2d(self.feat_ch * ch // 2, self.feat_ch * ch // 4, kernel_size=3, padding=1)
        # else:
        #     self.conv7 = nn.Conv2d(self.feat_ch * ch, self.feat_ch * ch // 2, kernel_size=3, padding=1)

        # if self.use_multiple_outputs:
        #     self.conv8_1 = nn.Conv2d(self.feat_ch * ch // 4, self.feat_ch * ch // 8, kernel_size=3, padding=1)
        #     self.conv8_2 = nn.Conv2d(self.feat_ch * ch // 4, self.feat_ch * ch // 8, kernel_size=3, padding=1)
        # else:
        #     self.conv8 = nn.Conv2d(self.feat_ch * ch // 2, self.feat_ch * ch // 4, kernel_size=3, padding=1)
                                       
        # if self.use_multiple_outputs:
        #     self.conv_final_1 = nn.Conv2d(self.feat_ch * ch // 8, self.output_nc // 2, kernel_size=3, padding=1)
        #     self.conv_final_2 = nn.Conv2d(self.feat_ch * ch // 8, self.output_nc // 2, kernel_size=3, padding=1)
        # else:
        #     self.conv_final = nn.Conv2d(self.feat_ch * ch // 4, self.output_nc, kernel_size=3, padding=1)
        self.conv_final = Conv(self.feat_ch * ch, self.output_nc, kernel_size=3, padding=1)

    def forward(self, merged_input, layers=[], encode_only=False, return_decomp=False):
        """
        Args:
            merged_input: [B, 1+ref_ch, H, W] - source + reference(s)
            layers: list of layer indices for encode_only mode
            encode_only: if True, return intermediate features
            return_decomp: if True, return style decomposition dict for loss computation
        """
        x = merged_input[:, :1, ...]
        ref = merged_input[:, 1:, ...]

        if self.is_3d:
            ref = ref.permute(0, 1, 4, 2, 3)
            style_guidance_raw = F.interpolate(ref, scale_factor=1/16, mode='trilinear', align_corners=False)
            style_guidance_raw = style_guidance_raw.permute(0, 1, 3, 4, 2)
            source_low = F.interpolate(x.permute(0, 1, 4, 2, 3), scale_factor=1/16, mode='trilinear', align_corners=False)
            source_low = source_low.permute(0, 1, 3, 4, 2)
        else:
            style_guidance_raw = F.interpolate(ref, scale_factor=1/16, mode='nearest')
            source_low = F.interpolate(x, scale_factor=1/16, mode='nearest')

        # Apply Common-Private Style Router if enabled
        decomp_dict = None
        if self.use_style_router and self.use_multiple_outputs and ref.shape[1] >= 2:
            style_b = style_guidance_raw[:, :1, ...]
            style_c = style_guidance_raw[:, 1:2, ...]

            # Route through Common-Private decomposition
            style_b_fused, style_c_fused, decomp_dict = self.style_router(
                style_b, style_c, source_low
            )

            # Reconstruct style_guidance_1 with fused styles
            style_guidance_1 = torch.cat([style_b_fused, style_c_fused], dim=1)
        else:
            style_guidance_1 = style_guidance_raw

        feats = []
        feat0 = self.conv0(x, style_guidance_1) # [1, feat_ch, H, W]
        feat1 = self.conv11(feat0, style_guidance_1) # [1, feat_ch, H/2, W/2]
        feat1 = self.conv12(feat1, style_guidance_1) # [1, feat_ch, H/2, W/2]
        feat2 = self.conv21(feat1, style_guidance_1) # [1, feat_ch, H/4, W/4]
        feat2 = self.conv22(feat2, style_guidance_1) # [1, feat_ch, H/4, W/4]
        feat3 = self.conv31(feat2, style_guidance_1) # [1, feat_ch, H/4, W/4] #TODO: 메모리 많아서 뺌.
        feat3 = self.conv32(feat3, style_guidance_1) # [1, feat_ch, H/4, W/4]
        feat4 = self.conv41(feat3 + feat2, style_guidance_1)# [1, feat_ch, H/2, W/2] #TODO: 원래 intput feat3 + feat2
        feat4 = self.conv42(feat4, style_guidance_1)        # [1, feat_ch, H/2, W/2]
        feat5 = self.conv51(feat4 + feat1, style_guidance_1)# [1, feat_ch, H, W]
        feat5 = self.conv52(feat5, style_guidance_1)        # [1, feat_ch, H, W]
        feat6 = self.conv6(feat5 + feat0, style_guidance_1) # [1, feat_ch, H, W]

        # Separate style layers: 채널을 완전히 분리해서 각각 독립적으로 처리
        if self.use_separate_style_layers and self.use_multiple_outputs:
            # feat6를 반으로 분리
            feat6_1, feat6_2 = torch.chunk(feat6, chunks=2, dim=1)  # 각 [B, feat_ch, H, W]
            # style도 분리
            style_1 = style_guidance_1[:, :1, ...]  # [B, 1, ...]
            style_2 = style_guidance_1[:, 1:, ...]  # [B, 1, ...]

            # conv7: 각각 독립적으로 처리
            feat7_1 = self.conv7_1(feat6_1, style_1)  # [B, feat_ch, H, W]
            feat7_2 = self.conv7_2(feat6_2, style_2)  # [B, feat_ch, H, W]

            # conv8: 각각 독립적으로 처리
            feat8_1 = self.conv8_1(feat7_1, style_1)  # [B, feat_ch, H, W]
            feat8_2 = self.conv8_2(feat7_2, style_2)  # [B, feat_ch, H, W]

            # 각각 conv_final + tanh
            out_1 = torch.tanh(self.conv_final_1(feat8_1))  # [B, output_nc//2, H, W]
            out_2 = torch.tanh(self.conv_final_2(feat8_2))  # [B, output_nc//2, H, W]

            # 마지막에 합쳐서 반환
            out = torch.cat((out_1, out_2), dim=1)  # [B, output_nc, H, W]
        else:
            out = self.conv_final(feat6)
            out = torch.tanh(out)

        if encode_only:
            layers_dict = {0: feat0, 1: feat1, 2: feat2, 3: feat3, 4: feat4, 5: feat5, 6: feat6}
            if self.use_separate_style_layers and self.use_multiple_outputs:
                layers_dict[7] = torch.cat((feat7_1, feat7_2), dim=1)
                layers_dict[8] = torch.cat((feat8_1, feat8_2), dim=1)
            return [layers_dict[i] for i in layers]

        if return_decomp and decomp_dict is not None:
            return out, decomp_dict
        return out


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
            
        # self.batch_norm = nn.BatchNorm2d(feat_ch, affine=False)
        self.normalize = Norm(feat_ch, affine=False)
        # self.batch_norm = nn.SyncBatchNorm(feat_ch, affine=False)

        nhidden = 512

        if ch == 1:
            self.mlp_shared = nn.Sequential( 
                Conv(1, nhidden, kernel_size=3, padding=1),
                nn.ReLU())
            self.mlp_gamma = Conv(nhidden, feat_ch, kernel_size=3, padding=1)
            self.mlp_beta = Conv(nhidden, feat_ch, kernel_size=3, padding=1)
        elif ch == 2:
            self.mlp_shared = nn.Sequential( 
                Conv(1, nhidden, kernel_size=3, padding=1),
                nn.ReLU())
            self.mlp_gamma = Conv(nhidden, feat_ch//2, kernel_size=3, padding=1)
            self.mlp_beta = Conv(nhidden, feat_ch//2, kernel_size=3, padding=1)

            self.mlp_shared_2 = nn.Sequential(
                Conv(1, nhidden, kernel_size=3, padding=1),
                nn.ReLU())
            self.mlp_gamma_2 = Conv(nhidden, feat_ch//2, kernel_size=3, padding=1)
            self.mlp_beta_2 = Conv(nhidden, feat_ch//2, kernel_size=3, padding=1)

        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.randomize_noise = True
        self.noise_strength = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x, style): # style: [1, 64, 1, 1]
        
        if self.downsample:
            original_size = x.size()
            if x.size()[2:] != original_size[2:]:  # If the size has changed
                x = F.interpolate(x, size=original_size[2:], mode='bilinear', align_corners=True)
            x = self.down(x)
        elif self.upsample:
            x = self.up(x)
            original_size = x.size()
            if x.size()[2:] != original_size[2:]:  # If the size has changed
                x = F.interpolate(x, size=original_size[2:], mode='bilinear', align_corners=True)
        else:
            x = self.conv(x)
        
        if style.shape[1] == 1:
            x = self.normalize(x)
        elif style.shape[1] == 2:
            x1, x2 = torch.chunk(x, chunks=2, dim=1)  # 각 부분 [B, C/2, H, W]
            x1 = self.normalize(x1)
            x2 = self.normalize(x2)
            x = torch.cat((x1, x2), dim=1)
        
        # # Add noise
        if self.randomize_noise:
            noise = torch.randn_like(x) * self.noise_strength
        else:
            noise = torch.zeros_like(x) * self.noise_strength
        
        x = x + noise
        if self.style_denorm:
            # 1. style을 x의 크기와 맞게 interpolation 
            mode = 'trilinear' if self.is_3d else 'nearest'
            style = F.interpolate(style, size=x.size()[2:], mode=mode) # TODO: bilinear 도 시도
            # 2. style을 x의 C과 맞게 mlp
            if style.shape[1] == 1:
                actv = self.mlp_shared(style)
                gamma = self.mlp_gamma(actv)
                beta = self.mlp_beta(actv)
            elif style.shape[1] == 2:
                actv1 = self.mlp_shared(style[:, :1, :, :])
                gamma = self.mlp_gamma(actv1) # B, 256, f_H, f_W
                beta = self.mlp_beta(actv1) # B, 256, f_H, f_W
                actv_2 = self.mlp_shared_2(style[:, 1:, :, :])
                gamma_2 = self.mlp_gamma_2(actv_2) 
                beta_2 = self.mlp_beta_2(actv_2) 
                gamma = torch.cat((gamma, gamma_2), dim=1) # B, 512, f_H, f_W
                beta = torch.cat((beta, beta_2), dim=1)

            # 3. 거기서 감마 베타 나눠서 denorm
            x = x * gamma + beta
        
        # activation (LeakyReLU)
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