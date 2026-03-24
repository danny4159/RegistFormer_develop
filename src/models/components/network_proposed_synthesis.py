import torch
import torch.nn as nn
import torch.nn.functional as F

def get_layer_by_dim(is_3d):
    dim = 3 if is_3d else 2
    Conv = getattr(nn, f'Conv{dim}d')
    Norm = getattr(nn, f'InstanceNorm{dim}d')
    return Conv, Norm, dim


class HighPassFilter(nn.Module):
    """Laplacian-based high-pass filter for extracting texture/style information.

    High-frequency components capture texture and local patterns while
    removing global structure information.
    """
    def __init__(self, is_3d=False):
        super().__init__()
        self.is_3d = is_3d

        # Laplacian kernel for edge/texture detection
        if is_3d:
            # 3D Laplacian kernel
            kernel = torch.tensor([
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
            ], dtype=torch.float32)
            kernel = kernel.view(1, 1, 3, 3, 3)
        else:
            # 2D Laplacian kernel
            kernel = torch.tensor([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ], dtype=torch.float32)
            kernel = kernel.view(1, 1, 3, 3)

        self.register_buffer('kernel', kernel)

    def forward(self, x):
        """Apply high-pass filter to input.

        Args:
            x: Input tensor [B, C, H, W] or [B, C, D, H, W]

        Returns:
            High-frequency components of x
        """
        B, C = x.shape[:2]

        if self.is_3d:
            # Process each channel separately
            x_reshaped = x.view(B * C, 1, *x.shape[2:])
            high_freq = F.conv3d(x_reshaped, self.kernel, padding=1)
            high_freq = high_freq.view(B, C, *x.shape[2:])
        else:
            x_reshaped = x.view(B * C, 1, *x.shape[2:])
            high_freq = F.conv2d(x_reshaped, self.kernel, padding=1)
            high_freq = high_freq.view(B, C, *x.shape[2:])

        return high_freq


class LocalStatistics(nn.Module):
    """Extract local mean and std as style features.

    Local statistics capture regional texture characteristics
    that are useful for style representation.
    """
    def __init__(self, kernel_size=5, is_3d=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.is_3d = is_3d
        self.padding = kernel_size // 2

    def forward(self, x):
        """Compute local mean and standard deviation.

        Args:
            x: Input tensor [B, C, H, W] or [B, C, D, H, W]

        Returns:
            Tuple of (local_mean, local_std)
        """
        if self.is_3d:
            # Use average pooling with same padding for local mean
            local_mean = F.avg_pool3d(x, self.kernel_size, stride=1, padding=self.padding)
            # Local variance: E[X^2] - E[X]^2
            local_sq_mean = F.avg_pool3d(x ** 2, self.kernel_size, stride=1, padding=self.padding)
        else:
            local_mean = F.avg_pool2d(x, self.kernel_size, stride=1, padding=self.padding)
            local_sq_mean = F.avg_pool2d(x ** 2, self.kernel_size, stride=1, padding=self.padding)

        local_var = local_sq_mean - local_mean ** 2
        local_std = torch.sqrt(local_var.clamp(min=1e-8))

        return local_mean, local_std


class CommonPrivateStyleRouter(nn.Module):
    """Common-Private Local Style Fusion (CPLSF) block.

    Decomposes each reference style into:
    - Common style: shared texture/style across references (e.g., MRI acquisition characteristics)
    - Private style: reference-specific style (e.g., individual anatomy)

    Only the common style is exchanged between references with conservative gating.
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

        # Style encoder: extracts features from reference
        # Input: 1 channel (single reference image)
        # Output: style_ch channels
        self.style_encoder = nn.Sequential(
            Conv(1, style_ch // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            Conv(style_ch // 2, style_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Frequency feature encoder (if using frequency prior)
        if use_freq_prior:
            # Input: 3 channels (high_freq, local_mean, local_std)
            self.freq_encoder = nn.Sequential(
                Conv(3, style_ch // 2, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                Conv(style_ch // 2, style_ch, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )
            # Learnable scale for freq_feat (conservative: start at 0.25, max 0.5)
            self.freq_scale = nn.Parameter(torch.tensor(0.25))

        # Common-Private decomposition networks
        # These learn to separate common (shared) from private (unique) components
        combined_ch = style_ch * 2 if use_freq_prior else style_ch

        self.common_head = nn.Sequential(
            Conv(combined_ch, style_ch, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            Conv(style_ch, style_ch, kernel_size=1),
        )

        self.private_head = nn.Sequential(
            Conv(combined_ch, style_ch, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            Conv(style_ch, style_ch, kernel_size=1),
        )

        # Conservative common fusion: average anchor + gated residual delta
        self.delta_fuser = nn.Sequential(
            Conv(style_ch * 2, style_ch, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            Conv(style_ch, style_ch, kernel_size=3, padding=1, groups=style_ch),
            Conv(style_ch, style_ch, kernel_size=1),
        )

        # Source-conditioned gating for common_shared fusion
        if use_source_gate:
            self.common_gate = nn.Sequential(
                Conv(style_ch + 1, style_ch, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                Conv(style_ch, style_ch, kernel_size=1),
                nn.Sigmoid(),
            )
            nn.init.constant_(self.common_gate[-2].bias, -2.0)

            # Branch-specific receiving gates: how much common_shared each branch absorbs
            self.gate_b = nn.Sequential(
                Conv(style_ch + 1, style_ch, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                Conv(style_ch, style_ch, kernel_size=1),
                nn.Sigmoid(),
            )
            self.gate_c = nn.Sequential(
                Conv(style_ch + 1, style_ch, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                Conv(style_ch, style_ch, kernel_size=1),
                nn.Sigmoid(),
            )
            nn.init.constant_(self.gate_b[-2].bias, -2.0)
            nn.init.constant_(self.gate_c[-2].bias, -2.0)

        # Learnable scaling for common absorption
        # Initialize to -4.0 so that initial alpha ≈ 0.25 * sigmoid(-4) ≈ 0.0045 (very conservative)
        param_shape = (1, style_ch, 1, 1, 1) if is_3d else (1, style_ch, 1, 1)
        self.alpha_b = nn.Parameter(torch.full(param_shape, -4.0))
        self.alpha_c = nn.Parameter(torch.full(param_shape, -4.0))

        # Project fused style to 1-channel for StyleConv compatibility
        self.style_out_b = Conv(style_ch, 1, kernel_size=1)
        self.style_out_c = Conv(style_ch, 1, kernel_size=1)

    def _prepare_input(self, style):
        if style.shape[1] == 1:
            return style
        return style.mean(dim=1, keepdim=True)

    def decompose_style(self, ref):
        """Decompose a single reference into common and private style components.

        Args:
            ref: Reference image [B, 1, H, W] or [B, 1, D, H, W]

        Returns:
            common: Common style features [B, style_ch, H, W]
            private: Private style features [B, style_ch, H, W]
        """
        # Extract base style features
        style_feat = self.style_encoder(ref)

        if self.use_freq_prior:
            # Extract frequency-based features
            high_freq = self.high_pass(ref)
            local_mean, local_std = self.local_stats(ref)
            freq_input = torch.cat([high_freq, local_mean, local_std], dim=1)
            freq_feat = self.freq_encoder(freq_input)

            # Apply conservative freq_scale (clamped to [0, 0.5])
            freq_scale = torch.clamp(self.freq_scale, min=0.0, max=0.5)
            combined = torch.cat([style_feat, freq_scale * freq_feat], dim=1)
        else:
            combined = style_feat

        # Decompose into common and private
        common = self.common_head(combined)
        private = self.private_head(combined)

        return common, private

    def forward(self, ref_b, ref_c, source_feat=None):
        """Process two references and return branch-specific fused styles.

        Args:
            ref_b: First reference image [B, 1, H, W]
            ref_c: Second reference image [B, 1, H, W]
            source_feat: Source features for gating [B, 1, H, W] (optional)

        Returns:
            style_b_map: Fused style for branch b [B, 1, H, W]
            style_c_map: Fused style for branch c [B, 1, H, W]
            decomp_dict: Dictionary containing decomposition info for loss computation
        """
        ref_b = self._prepare_input(ref_b)
        ref_c = self._prepare_input(ref_c)

        # Decompose each reference into common and private
        common_b, private_b = self.decompose_style(ref_b)
        common_c, private_c = self.decompose_style(ref_c)

        # Conservative common fusion: average + gated delta
        common_avg = 0.5 * (common_b + common_c)
        delta = self.delta_fuser(torch.cat([
            common_b - common_avg,
            common_c - common_avg,
        ], dim=1))

        # Resize source_feat if needed
        if self.use_source_gate and source_feat is not None:
            source_feat = self._prepare_input(source_feat)
            if source_feat.shape[2:] != common_avg.shape[2:]:
                if self.is_3d:
                    source_feat = F.interpolate(
                        source_feat,
                        size=common_avg.shape[2:],
                        mode='trilinear',
                        align_corners=False,
                    )
                else:
                    source_feat = F.interpolate(source_feat, size=common_avg.shape[2:], mode='nearest')
            g_common = self.common_gate(torch.cat([common_avg, source_feat], dim=1))
        else:
            g_common = torch.sigmoid(common_avg) * 0.1  # Conservative default

        common_shared = common_avg + g_common * delta

        # Branch-specific receiving gates: how much common_shared each branch absorbs
        if self.use_source_gate and source_feat is not None:
            gate_b = self.gate_b(torch.cat([common_b, source_feat], dim=1))
            gate_c = self.gate_c(torch.cat([common_c, source_feat], dim=1))
        else:
            gate_b = torch.full_like(common_shared, 0.1)
            gate_c = torch.full_like(common_shared, 0.1)

        # Final fused style per branch: private + alpha * (gate * common_shared)
        # s_b = p_b + alpha_b * (g_b * c_shared)
        # Cap alpha to [0, 0.25] range via 0.25 * sigmoid
        alpha_b = 0.25 * torch.sigmoid(self.alpha_b)
        alpha_c = 0.25 * torch.sigmoid(self.alpha_c)
        style_b_feat = private_b + alpha_b * (gate_b * common_shared)
        style_c_feat = private_c + alpha_c * (gate_c * common_shared)

        # Project to 1-channel for StyleConv compatibility
        style_b_map = self.style_out_b(style_b_feat)
        style_c_map = self.style_out_c(style_c_feat)

        decomp_dict = {
            'common_b': common_b,
            'common_c': common_c,
            'private_b': private_b,
            'private_c': private_c,
            'common_shared': common_shared,
            'g_common': g_common,
            'g_b': gate_b,
            'g_c': gate_c,
            'style_b_feat': style_b_feat,
            'style_c_feat': style_c_feat,
        }

        return style_b_map, style_c_map, decomp_dict


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
            # Style router parameters
            self.use_style_router = kwargs.get('use_style_router', False)
            self.style_router_ch = kwargs.get('style_router_ch', 64)
            self.use_freq_prior = kwargs.get('use_freq_prior', True)
            self.use_source_gate = kwargs.get('use_source_gate', True)

        except KeyError as e:
            raise ValueError(f"Missing required parameter: {str(e)}")

        Conv, _, _ = get_layer_by_dim(self.is_3d)

        # Initialize style router if enabled
        if self.use_style_router and self.use_multiple_outputs:
            self.style_router = CommonPrivateStyleRouter(
                style_ch=self.style_router_ch,
                use_freq_prior=self.use_freq_prior,
                use_source_gate=self.use_source_gate,
                is_3d=self.is_3d,
            )

        ch = 2 if self.use_multiple_outputs else 1
        # When using style router, we still use ch=2 for StyleConv
        # because fused_style = cat(style_b_map, style_c_map) has 2 channels
        shared_style_ch = ch
        self.use_branch_style_layers = self.use_multiple_outputs and (
            self.use_separate_style_layers or self.use_style_router
        )

        guide_hidden = max(1, int(self.feat_ch / 8))
        self.guide_net = nn.Sequential(
            Conv(self.input_nc, guide_hidden, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            Conv(guide_hidden, guide_hidden, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            Conv(guide_hidden, 1, kernel_size=3, stride=1, padding=1),
        )
        
        # 일부분에만 style_denorm -> 21, 22, 31, 32에만 적용 #TODO: feat_ch에 ref ch만큼 배수로
        self.conv0 = StyleConv(self.input_nc, self.feat_ch * ch, kernel_size=3,
                                                 activate=True, demodulate=self.demodulate, ch=shared_style_ch, is_3d=self.is_3d)
        self.conv11 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=True, activate=True, demodulate=self.demodulate, ch=shared_style_ch, is_3d=self.is_3d)
        self.conv12 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=self.demodulate, ch=shared_style_ch, is_3d=self.is_3d)
        self.conv21 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=True, activate=True, demodulate=self.demodulate, ch=shared_style_ch, is_3d=self.is_3d)
        self.conv22 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=self.demodulate, ch=shared_style_ch, is_3d=self.is_3d)
        self.conv31 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=self.demodulate, ch=shared_style_ch, is_3d=self.is_3d)
        self.conv32 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=self.demodulate, ch=shared_style_ch, is_3d=self.is_3d)
        self.conv41 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3, #feat_ch *4는 변치않게
                                upsample=True, activate=True, demodulate=self.demodulate, ch=shared_style_ch, is_3d=self.is_3d)
        self.conv42 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                upsample=False, activate=True, demodulate=self.demodulate, ch=shared_style_ch, is_3d=self.is_3d)
        self.conv51 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                upsample=True, activate=True, demodulate=self.demodulate, ch=shared_style_ch, is_3d=self.is_3d)
        self.conv52 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                upsample=False, activate=True, demodulate=self.demodulate, ch=shared_style_ch, is_3d=self.is_3d)
        self.conv6 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                activate=True, demodulate=self.demodulate, ch=shared_style_ch, is_3d=self.is_3d) # 이걸 빠트렸었어.
                                # activate=False, demodulate=self.demodulate)

        # Separate style layers: 채널을 완전히 분리해서 각각 독립적으로 style 적용
        if self.use_branch_style_layers:
            # Split the synthesis path after conv22 so mid/late synthesis is branch-specific.
            self.conv31_1 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                      downsample=False, activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d)
            self.conv31_2 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                      downsample=False, activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d)
            self.conv32_1 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                      downsample=False, activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d)
            self.conv32_2 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                      downsample=False, activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d)
            self.conv41_1 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                      upsample=True, activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d)
            self.conv41_2 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                      upsample=True, activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d)
            self.conv42_1 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                      upsample=False, activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d)
            self.conv42_2 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                      upsample=False, activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d)
            self.conv51_1 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                      upsample=True, activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d)
            self.conv51_2 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                      upsample=True, activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d)
            self.conv52_1 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                      upsample=False, activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d)
            self.conv52_2 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                      upsample=False, activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d)
            self.conv6_1 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d)
            self.conv6_2 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d)
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

    def forward(self, merged_input, layers=[], encode_only=False):

        x = merged_input[:, :1, ...]
        ref = merged_input[:, 1:, ...]

        decomp_dict = None  # Will be populated if using style router

        if self.is_3d:
            ref = ref.permute(0, 1, 4, 2, 3)
            style_guidance_raw = F.interpolate(ref, scale_factor=1/16, mode='trilinear', align_corners=False)
            style_guidance_raw = style_guidance_raw.permute(0, 1, 3, 4, 2)

            x_low = x.permute(0, 1, 4, 2, 3)
            source_low = F.interpolate(x_low, scale_factor=1/16, mode='trilinear', align_corners=False)
            source_low = source_low.permute(0, 1, 3, 4, 2)
        else:
            style_guidance_raw = F.interpolate(ref, scale_factor=1/16, mode='nearest') # Final #TODO: sliding infer로 줄어든만큼 이것도 줄여줘야해. 8정도가 적절할듯
            source_low = F.interpolate(x, scale_factor=1/16, mode='nearest')

        # Default: use raw style guidance (for non-router case)
        fused_style = style_guidance_raw  # [B, 2, H, W] when use_multiple_outputs
        style_b_map = style_guidance_raw[:, :1, ...]
        style_c_map = style_guidance_raw[:, 1:2, ...] if style_guidance_raw.shape[1] > 1 else style_b_map

        # Apply style router if enabled
        if self.use_style_router and self.use_multiple_outputs and style_guidance_raw.shape[1] >= 2:
            ref_b = style_guidance_raw[:, :1, ...]
            ref_c = style_guidance_raw[:, 1:2, ...]
            content_hint = self.guide_net(source_low)

            # Router returns branch-specific fused styles
            style_b_map, style_c_map, decomp_dict = self.style_router(ref_b, ref_c, content_hint)

            # fused_style = cat(style_b_map, style_c_map) for StyleConv(ch=2)
            fused_style = torch.cat([style_b_map, style_c_map], dim=1)  # [B, 2, H, W]

        # style_guidance_1 = F.interpolate(ref, scale_factor=1/8, mode='bilinear', align_corners=True) # Ablation
        # style_guidance_1 = F.interpolate(ref, scale_factor=1/32, mode='bilinear', align_corners=True) # Ablation
        # style_guidance_1 = F.interpolate(ref, scale_factor=1/64, mode='bilinear', align_corners=True) # Ablation
        # style_guidance_1 = F.interpolate(ref, size=(1, 1), mode='nearest') # Ablation
        feats = []
        feat0 = self.conv0(x, fused_style) # [1, feat_ch*2, H, W]
        feat1 = self.conv11(feat0, fused_style) # [1, feat_ch*2, H/2, W/2]
        feat1 = self.conv12(feat1, fused_style) # [1, feat_ch*2, H/2, W/2]
        feat2 = self.conv21(feat1, fused_style) # [1, feat_ch*2, H/4, W/4]
        feat2 = self.conv22(feat2, fused_style) # [1, feat_ch*2, H/4, W/4]

        # Separate style layers: conv22 이후부터 branch-specific synthesis로 진행
        if self.use_branch_style_layers:
            feat0_1, feat0_2 = torch.chunk(feat0, chunks=2, dim=1)
            feat1_1, feat1_2 = torch.chunk(feat1, chunks=2, dim=1)
            feat2_1, feat2_2 = torch.chunk(feat2, chunks=2, dim=1)

            feat3_1 = self.conv31_1(feat2_1, style_b_map)
            feat3_1 = self.conv32_1(feat3_1, style_b_map)
            feat3_2 = self.conv31_2(feat2_2, style_c_map)
            feat3_2 = self.conv32_2(feat3_2, style_c_map)
            feat3 = torch.cat((feat3_1, feat3_2), dim=1)

            feat4_1 = self.conv41_1(feat3_1 + feat2_1, style_b_map)
            feat4_1 = self.conv42_1(feat4_1, style_b_map)
            feat4_2 = self.conv41_2(feat3_2 + feat2_2, style_c_map)
            feat4_2 = self.conv42_2(feat4_2, style_c_map)
            feat4 = torch.cat((feat4_1, feat4_2), dim=1)

            feat5_1 = self.conv51_1(feat4_1 + feat1_1, style_b_map)
            feat5_1 = self.conv52_1(feat5_1, style_b_map)
            feat5_2 = self.conv51_2(feat4_2 + feat1_2, style_c_map)
            feat5_2 = self.conv52_2(feat5_2, style_c_map)
            feat5 = torch.cat((feat5_1, feat5_2), dim=1)

            feat6_1 = self.conv6_1(feat5_1 + feat0_1, style_b_map)
            feat6_2 = self.conv6_2(feat5_2 + feat0_2, style_c_map)
            feat6 = torch.cat((feat6_1, feat6_2), dim=1)

            feat7_1 = self.conv7_1(feat6_1, style_b_map)
            feat7_2 = self.conv7_2(feat6_2, style_c_map)
            feat8_1 = self.conv8_1(feat7_1, style_b_map)
            feat8_2 = self.conv8_2(feat7_2, style_c_map)

            out_1 = torch.tanh(self.conv_final_1(feat8_1))
            out_2 = torch.tanh(self.conv_final_2(feat8_2))
            out = torch.cat((out_1, out_2), dim=1)
        else:
            feat3 = self.conv31(feat2, fused_style) # [1, feat_ch*2, H/4, W/4]
            feat3 = self.conv32(feat3, fused_style) # [1, feat_ch*2, H/4, W/4]
            feat4 = self.conv41(feat3 + feat2, fused_style) # [1, feat_ch*2, H/2, W/2]
            feat4 = self.conv42(feat4, fused_style)         # [1, feat_ch*2, H/2, W/2]
            feat5 = self.conv51(feat4 + feat1, fused_style) # [1, feat_ch*2, H, W]
            feat5 = self.conv52(feat5, fused_style)         # [1, feat_ch*2, H, W]
            feat6 = self.conv6(feat5 + feat0, fused_style)  # [1, feat_ch*2, H, W]
            out = self.conv_final(feat6)
            out = torch.tanh(out)

        if encode_only:
            layers_dict = {0: feat0, 1: feat1, 2: feat2, 3: feat3, 4: feat4, 5: feat5, 6: feat6}
            if self.use_branch_style_layers:
                layers_dict[7] = torch.cat((feat7_1, feat7_2), dim=1)
                layers_dict[8] = torch.cat((feat8_1, feat8_2), dim=1)
            return [layers_dict[i] for i in layers]

        # Return output and decomp_dict (decomp_dict is None if style router is not used)
        return out, decomp_dict


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
                if self.is_3d:
                    x = F.interpolate(x, size=original_size[2:], mode='trilinear', align_corners=False)
                else:
                    x = F.interpolate(x, size=original_size[2:], mode='bilinear', align_corners=False)
            x = self.down(x)
        elif self.upsample:
            x = self.up(x)
            original_size = x.size()
            if x.size()[2:] != original_size[2:]:  # If the size has changed
                if self.is_3d:
                    x = F.interpolate(x, size=original_size[2:], mode='trilinear', align_corners=False)
                else:
                    x = F.interpolate(x, size=original_size[2:], mode='bilinear', align_corners=False)
        else:
            x = self.conv(x)
        
        if style.shape[1] == 1:
            x = self.normalize(x)
        elif style.shape[1] == 2:
            x1, x2 = torch.chunk(x, chunks=2, dim=1)  # 각 부분 [B, C/2, H, W]
            x1 = self.normalize(x1)
            x2 = self.normalize(x2)
            x = torch.cat((x1, x2), dim=1)
        else:
            style = style.mean(dim=1, keepdim=True)
            x = self.normalize(x)
        
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
                actv1 = self.mlp_shared(style[:, :1, ...])
                gamma = self.mlp_gamma(actv1) # B, 256, f_H, f_W
                beta = self.mlp_beta(actv1) # B, 256, f_H, f_W
                actv_2 = self.mlp_shared_2(style[:, 1:2, ...])
                gamma_2 = self.mlp_gamma_2(actv_2) 
                beta_2 = self.mlp_beta_2(actv_2) 
                gamma = torch.cat((gamma, gamma_2), dim=1) # B, 512, f_H, f_W
                beta = torch.cat((beta, beta_2), dim=1)
            else:
                style = style.mean(dim=1, keepdim=True)
                actv = self.mlp_shared(style)
                gamma = self.mlp_gamma(actv)
                beta = self.mlp_beta(actv)

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