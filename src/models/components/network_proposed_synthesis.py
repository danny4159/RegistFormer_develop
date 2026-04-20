import torch
import torch.nn as nn
import torch.nn.functional as F

def get_layer_by_dim(is_3d):
    dim = 3 if is_3d else 2
    Conv = getattr(nn, f'Conv{dim}d')
    Norm = getattr(nn, f'InstanceNorm{dim}d')
    return Conv, Norm, dim

class ProposedSynthesisModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        try:
            self.input_nc = kwargs['input_nc']
            self.feat_ch = kwargs['feat_ch']
            self.output_nc = kwargs['output_nc']
            self.demodulate = kwargs['demodulate']
            self.use_multiple_outputs = kwargs.get('use_multiple_outputs', None)
            self.use_triple_outputs = kwargs.get('use_triple_outputs', False)
            self.is_3d = kwargs.get('is_3d', False)
            self.use_separate_style_layers = kwargs.get('use_separate_style_layers', False)
            self.noise_independent = kwargs.get('noise_independent', False)
            self.use_25d_style = kwargs.get('use_25d_style', False)
            self.ref_stack_size = kwargs.get('ref_stack_size', 3)
            self.z_agg_deep = kwargs.get('z_agg_deep', False)
            self.use_center_residual_style = kwargs.get('use_center_residual_style', False)

        except KeyError as e:
            raise ValueError(f"Missing required parameter: {str(e)}")

        Conv, _, _ = get_layer_by_dim(self.is_3d)

        # Determine channel multiplier based on output mode
        if self.use_triple_outputs:
            ch = 3
            self.num_style_streams = 3
        elif self.use_multiple_outputs:
            ch = 2
            self.num_style_streams = 2
        else:
            ch = 1
            self.num_style_streams = 1

        # 2.5D: compress ref stack [B, K, H, W] -> [B, 1, H, W] per style stream
        if self.use_25d_style:
            hidden = max(8, self.feat_ch // 8)
            if self.z_agg_deep:
                def _make_z_agg():
                    return nn.Sequential(
                        nn.Conv2d(self.ref_stack_size, hidden, kernel_size=3, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
                    )
            else:
                def _make_z_agg():
                    return nn.Sequential(
                        nn.Conv2d(self.ref_stack_size, hidden, kernel_size=3, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
                    )
            self.z_aggs = nn.ModuleList([_make_z_agg() for _ in range(self.num_style_streams)])

        self.guide_net = nn.Sequential(
            nn.Conv2d(self.input_nc, int(self.feat_ch / 8), kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(int(self.feat_ch / 8), int(self.feat_ch / 8), kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(int(self.feat_ch / 8), int(self.feat_ch / 8), kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 일부분에만 style_denorm -> 21, 22, 31, 32에만 적용 #TODO: feat_ch에 ref ch만큼 배수로
        _sc_kw = dict(activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d,
                      noise_independent=self.noise_independent,
                      use_center_residual_style=self.use_center_residual_style)
        self.conv0  = StyleConv(self.input_nc,    self.feat_ch * ch, kernel_size=3, **_sc_kw)
        self.conv11 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3, downsample=True,  **_sc_kw)
        self.conv12 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3, downsample=False, **_sc_kw)
        self.conv21 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3, downsample=True,  **_sc_kw)
        self.conv22 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3, downsample=False, **_sc_kw)
        self.conv31 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3, downsample=False, **_sc_kw)
        self.conv32 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3, downsample=False, **_sc_kw)
        self.conv41 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3, upsample=True,    **_sc_kw)
        self.conv42 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3, upsample=False,   **_sc_kw)
        self.conv51 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3, upsample=True,    **_sc_kw)
        self.conv52 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3, upsample=False,   **_sc_kw)
        self.conv6  = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3, **_sc_kw)

        # Separate style layers: 채널을 완전히 분리해서 각각 독립적으로 style 적용
        if self.use_separate_style_layers and self.use_triple_outputs:
            # 각 branch는 feat_ch 채널 (전체의 1/3)
            self.conv7_1 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d)
            self.conv7_2 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d)
            self.conv7_3 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d)
            self.conv8_1 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d)
            self.conv8_2 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d)
            self.conv8_3 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d)
            # 각각 독립적인 conv_final
            self.conv_final_1 = Conv(self.feat_ch, self.output_nc // 3, kernel_size=3, padding=1)
            self.conv_final_2 = Conv(self.feat_ch, self.output_nc // 3, kernel_size=3, padding=1)
            self.conv_final_3 = Conv(self.feat_ch, self.output_nc // 3, kernel_size=3, padding=1)
        elif self.use_separate_style_layers and self.use_multiple_outputs:
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

    def _aggregate_ref_stack(self, ref_all):
        """2.5D: [B, num_streams*K, H, W] -> [B, num_streams, H, W]"""
        if not self.use_25d_style:
            return ref_all
        expected_ch = self.num_style_streams * self.ref_stack_size
        assert ref_all.shape[1] == expected_ch, \
            f"Expected {expected_ch} ref channels, got {ref_all.shape[1]}"
        chunks = torch.split(ref_all, self.ref_stack_size, dim=1)
        style_maps = [z_agg(chunk) for chunk, z_agg in zip(chunks, self.z_aggs)]
        return torch.cat(style_maps, dim=1)

    def _split_center_and_ctx_ref(self, ref_all):
        """Option C-2: returns (ref_center, ref_ctx) both [B, num_streams, H, W].
        ref_center = center slice per stream; ref_ctx = z_agg output per stream.
        """
        expected_ch = self.num_style_streams * self.ref_stack_size
        assert ref_all.shape[1] == expected_ch, \
            f"Expected {expected_ch} ref channels, got {ref_all.shape[1]}"
        center_idx = self.ref_stack_size // 2
        chunks = torch.split(ref_all, self.ref_stack_size, dim=1)
        centers, ctxs = [], []
        for chunk, z_agg in zip(chunks, self.z_aggs):
            centers.append(chunk[:, center_idx:center_idx + 1])
            ctxs.append(z_agg(chunk))
        return torch.cat(centers, dim=1), torch.cat(ctxs, dim=1)

    def _make_style_guidance(self, ref):
        if self.is_3d:
            ref = ref.permute(0, 1, 4, 2, 3)
            sg = F.interpolate(ref, scale_factor=1/16, mode='trilinear', align_corners=False)
            return sg.permute(0, 1, 3, 4, 2)
        return F.interpolate(ref, scale_factor=1/16, mode='nearest')

    def forward(self, merged_input, layers=[], encode_only=False):

        x = merged_input[:, :1, ...]
        ref_all = merged_input[:, 1:, ...]

        use_c2 = self.use_25d_style and self.use_center_residual_style
        if use_c2:
            ref_center, ref_ctx = self._split_center_and_ctx_ref(ref_all)
            sg_center = self._make_style_guidance(ref_center)
            sg_ctx    = self._make_style_guidance(ref_ctx)
        else:
            ref = self._aggregate_ref_stack(ref_all)
            sg_center = self._make_style_guidance(ref)
            sg_ctx = None

        feats = []
        feat0 = self.conv0(x,            sg_center, sg_ctx)
        feat1 = self.conv11(feat0,        sg_center, sg_ctx)
        feat1 = self.conv12(feat1,        sg_center, sg_ctx)
        feat2 = self.conv21(feat1,        sg_center, sg_ctx)
        feat2 = self.conv22(feat2,        sg_center, sg_ctx)
        feat3 = self.conv31(feat2,        sg_center, sg_ctx)
        feat3 = self.conv32(feat3,        sg_center, sg_ctx)
        feat4 = self.conv41(feat3 + feat2, sg_center, sg_ctx)
        feat4 = self.conv42(feat4,         sg_center, sg_ctx)
        feat5 = self.conv51(feat4 + feat1, sg_center, sg_ctx)
        feat5 = self.conv52(feat5,         sg_center, sg_ctx)
        feat6 = self.conv6(feat5 + feat0,  sg_center, sg_ctx)

        # Separate style layers: 채널을 완전히 분리해서 각각 독립적으로 처리
        if self.use_separate_style_layers and self.use_triple_outputs:
            # feat6를 3등분으로 분리
            feat6_1, feat6_2, feat6_3 = torch.chunk(feat6, chunks=3, dim=1)  # 각 [B, feat_ch, H, W]
            # style도 분리
            style_1 = sg_center[:, :1, ...]
            style_2 = sg_center[:, 1:2, ...]
            style_3 = sg_center[:, 2:3, ...]
            ctx_1 = sg_ctx[:, :1, ...] if sg_ctx is not None else None
            ctx_2 = sg_ctx[:, 1:2, ...] if sg_ctx is not None else None
            ctx_3 = sg_ctx[:, 2:3, ...] if sg_ctx is not None else None

            # conv7: 각각 독립적으로 처리
            feat7_1 = self.conv7_1(feat6_1, style_1, ctx_1)
            feat7_2 = self.conv7_2(feat6_2, style_2, ctx_2)
            feat7_3 = self.conv7_3(feat6_3, style_3, ctx_3)

            # conv8: 각각 독립적으로 처리
            feat8_1 = self.conv8_1(feat7_1, style_1, ctx_1)
            feat8_2 = self.conv8_2(feat7_2, style_2, ctx_2)
            feat8_3 = self.conv8_3(feat7_3, style_3, ctx_3)

            # 각각 conv_final + tanh
            out_1 = torch.tanh(self.conv_final_1(feat8_1))  # [B, output_nc//3, H, W]
            out_2 = torch.tanh(self.conv_final_2(feat8_2))  # [B, output_nc//3, H, W]
            out_3 = torch.tanh(self.conv_final_3(feat8_3))  # [B, output_nc//3, H, W]

            # 마지막에 합쳐서 반환
            out = torch.cat((out_1, out_2, out_3), dim=1)  # [B, output_nc, H, W]
        elif self.use_separate_style_layers and self.use_multiple_outputs:
            # feat6를 반으로 분리
            feat6_1, feat6_2 = torch.chunk(feat6, chunks=2, dim=1)  # 각 [B, feat_ch, H, W]
            # style도 분리
            style_1 = sg_center[:, :1, ...]
            style_2 = sg_center[:, 1:, ...]
            ctx_1 = sg_ctx[:, :1, ...] if sg_ctx is not None else None
            ctx_2 = sg_ctx[:, 1:, ...] if sg_ctx is not None else None

            # conv7: 각각 독립적으로 처리
            feat7_1 = self.conv7_1(feat6_1, style_1, ctx_1)
            feat7_2 = self.conv7_2(feat6_2, style_2, ctx_2)

            # conv8: 각각 독립적으로 처리
            feat8_1 = self.conv8_1(feat7_1, style_1, ctx_1)
            feat8_2 = self.conv8_2(feat7_2, style_2, ctx_2)

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
            if self.use_separate_style_layers and self.use_triple_outputs:
                layers_dict[7] = torch.cat((feat7_1, feat7_2, feat7_3), dim=1)
                layers_dict[8] = torch.cat((feat8_1, feat8_2, feat8_3), dim=1)
            elif self.use_separate_style_layers and self.use_multiple_outputs:
                layers_dict[7] = torch.cat((feat7_1, feat7_2), dim=1)
                layers_dict[8] = torch.cat((feat8_1, feat8_2), dim=1)
            return [layers_dict[i] for i in layers]

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
                 is_3d=False,
                 noise_independent=False,
                 use_center_residual_style=False):

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
        self.noise_independent = noise_independent
        self.use_center_residual_style = use_center_residual_style

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

        self.ch = ch
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
        elif ch == 3:
            self.mlp_shared = nn.Sequential(
                Conv(1, nhidden, kernel_size=3, padding=1),
                nn.ReLU())
            self.mlp_gamma = Conv(nhidden, feat_ch//3, kernel_size=3, padding=1)
            self.mlp_beta = Conv(nhidden, feat_ch//3, kernel_size=3, padding=1)

            self.mlp_shared_2 = nn.Sequential(
                Conv(1, nhidden, kernel_size=3, padding=1),
                nn.ReLU())
            self.mlp_gamma_2 = Conv(nhidden, feat_ch//3, kernel_size=3, padding=1)
            self.mlp_beta_2 = Conv(nhidden, feat_ch//3, kernel_size=3, padding=1)

            self.mlp_shared_3 = nn.Sequential(
                Conv(1, nhidden, kernel_size=3, padding=1),
                nn.ReLU())
            self.mlp_gamma_3 = Conv(nhidden, feat_ch//3, kernel_size=3, padding=1)
            self.mlp_beta_3 = Conv(nhidden, feat_ch//3, kernel_size=3, padding=1)

        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if self.use_center_residual_style:
            gate_hidden = max(16, feat_ch // 8)
            self.gate_net = nn.Sequential(
                Conv(2 * ch, gate_hidden, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                Conv(gate_hidden, feat_ch, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )
            self.ctx_res_scale = nn.Parameter(torch.zeros(1))

        self.randomize_noise = True
        # noise_independent=True: chunk별로 개별 noise_strength (독립적)
        # noise_independent=False: 기존처럼 하나의 noise_strength (cat된 상태에서 noise 주입)
        if not self.noise_independent:
            self.noise_strength = nn.Parameter(torch.zeros(1), requires_grad=True)
        else:
            if ch == 1:
                self.noise_strength = nn.Parameter(torch.zeros(1), requires_grad=True)
            elif ch == 2:
                self.noise_strength_1 = nn.Parameter(torch.zeros(1), requires_grad=True)
                self.noise_strength_2 = nn.Parameter(torch.zeros(1), requires_grad=True)
            elif ch == 3:
                self.noise_strength_1 = nn.Parameter(torch.zeros(1), requires_grad=True)
                self.noise_strength_2 = nn.Parameter(torch.zeros(1), requires_grad=True)
                self.noise_strength_3 = nn.Parameter(torch.zeros(1), requires_grad=True)

    def _style_to_gamma_beta(self, style):
        """style [B, ch, H, W] -> gamma, beta [B, feat_ch, H, W]"""
        if style.shape[1] == 1:
            actv = self.mlp_shared(style)
            gamma = self.mlp_gamma(actv)
            beta = self.mlp_beta(actv)
        elif style.shape[1] == 2:
            actv1 = self.mlp_shared(style[:, :1, ...])
            actv2 = self.mlp_shared_2(style[:, 1:2, ...])
            gamma = torch.cat((self.mlp_gamma(actv1), self.mlp_gamma_2(actv2)), dim=1)
            beta = torch.cat((self.mlp_beta(actv1), self.mlp_beta_2(actv2)), dim=1)
        elif style.shape[1] == 3:
            actv1 = self.mlp_shared(style[:, :1, ...])
            actv2 = self.mlp_shared_2(style[:, 1:2, ...])
            actv3 = self.mlp_shared_3(style[:, 2:3, ...])
            gamma = torch.cat((self.mlp_gamma(actv1), self.mlp_gamma_2(actv2), self.mlp_gamma_3(actv3)), dim=1)
            beta = torch.cat((self.mlp_beta(actv1), self.mlp_beta_2(actv2), self.mlp_beta_3(actv3)), dim=1)
        else:
            raise ValueError(f"Unsupported style channels: {style.shape[1]}")
        return gamma, beta

    def forward(self, x, style_center, style_ctx=None): # style_center: [B, ch, H, W]
        
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
        
        if not self.noise_independent:
            # 기존 방식: normalize -> cat -> noise 한꺼번에
            if style_center.shape[1] == 1:
                x = self.normalize(x)
            elif style_center.shape[1] == 2:
                x1, x2 = torch.chunk(x, chunks=2, dim=1)  # 각 부분 [B, C/2, H, W]
                x1 = self.normalize(x1)
                x2 = self.normalize(x2)
                x = torch.cat((x1, x2), dim=1)
            elif style_center.shape[1] == 3:
                x1, x2, x3 = torch.chunk(x, chunks=3, dim=1)  # 각 부분 [B, C/3, H, W]
                x1 = self.normalize(x1)
                x2 = self.normalize(x2)
                x3 = self.normalize(x3)
                x = torch.cat((x1, x2, x3), dim=1)

            # Add noise
            if self.randomize_noise:
                noise = torch.randn_like(x) * self.noise_strength
            else:
                noise = torch.zeros_like(x) * self.noise_strength

            x = x + noise
        else:
            # 새로운 방식: normalize -> noise 개별 주입 -> cat
            if style_center.shape[1] == 1:
                x = self.normalize(x)
                if self.randomize_noise:
                    noise = torch.randn_like(x) * self.noise_strength
                else:
                    noise = torch.zeros_like(x) * self.noise_strength
                x = x + noise
            elif style_center.shape[1] == 2:
                x1, x2 = torch.chunk(x, chunks=2, dim=1)  # 각 부분 [B, C/2, H, W]
                x1 = self.normalize(x1)
                x2 = self.normalize(x2)
                # noise 개별 주입
                if self.randomize_noise:
                    noise1 = torch.randn_like(x1) * self.noise_strength_1
                    noise2 = torch.randn_like(x2) * self.noise_strength_2
                else:
                    noise1 = torch.zeros_like(x1) * self.noise_strength_1
                    noise2 = torch.zeros_like(x2) * self.noise_strength_2
                x1 = x1 + noise1
                x2 = x2 + noise2
                x = torch.cat((x1, x2), dim=1)
            elif style_center.shape[1] == 3:
                x1, x2, x3 = torch.chunk(x, chunks=3, dim=1)  # 각 부분 [B, C/3, H, W]
                x1 = self.normalize(x1)
                x2 = self.normalize(x2)
                x3 = self.normalize(x3)
                # noise 개별 주입
                if self.randomize_noise:
                    noise1 = torch.randn_like(x1) * self.noise_strength_1
                    noise2 = torch.randn_like(x2) * self.noise_strength_2
                    noise3 = torch.randn_like(x3) * self.noise_strength_3
                else:
                    noise1 = torch.zeros_like(x1) * self.noise_strength_1
                    noise2 = torch.zeros_like(x2) * self.noise_strength_2
                    noise3 = torch.zeros_like(x3) * self.noise_strength_3
                x1 = x1 + noise1
                x2 = x2 + noise2
                x3 = x3 + noise3
                x = torch.cat((x1, x2, x3), dim=1)
        if self.style_denorm:
            mode = 'trilinear' if self.is_3d else 'nearest'
            style_center = F.interpolate(style_center, size=x.size()[2:], mode=mode)

            if self.use_center_residual_style and style_ctx is not None:
                style_ctx = F.interpolate(style_ctx, size=x.size()[2:], mode=mode)
                gamma_center, beta_center = self._style_to_gamma_beta(style_center)
                gamma_ctx, beta_ctx = self._style_to_gamma_beta(style_ctx)
                gate = self.gate_net(torch.cat((style_center, style_ctx), dim=1))
                scale = torch.tanh(self.ctx_res_scale)
                gamma = gamma_center + scale * gate * (gamma_ctx - gamma_center)
                beta = beta_center + scale * gate * (beta_ctx - beta_center)
            else:
                gamma, beta = self._style_to_gamma_beta(style_center)

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