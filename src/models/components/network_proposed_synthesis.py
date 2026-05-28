import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_layer_by_dim(is_3d):
    dim = 3 if is_3d else 2
    Conv = getattr(nn, f'Conv{dim}d')
    Norm = getattr(nn, f'InstanceNorm{dim}d')
    return Conv, Norm, dim


def spatial_zscore(x, eps=1e-6):
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), keepdim=True)
    return (x - mean) / (std + eps)


def sobel_mag(x, eps=1e-6):
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=device, dtype=dtype).view(1,1,3,3).repeat(C,1,1,1)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=device, dtype=dtype).view(1,1,3,3).repeat(C,1,1,1)
    gx = F.conv2d(x, kx, padding=1, groups=C)
    gy = F.conv2d(x, ky, padding=1, groups=C)
    return torch.sqrt(gx * gx + gy * gy + eps)


class LowResSliceSelector(nn.Module):
    def __init__(self, in_ch, ref_stack_size, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, ref_stack_size, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


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

            self.z_select_mode        = kwargs.get('z_select_mode', 'conv')
            self.z_select_scale       = int(kwargs.get('z_select_scale', 16))
            self.z_select_hidden      = int(kwargs.get('z_select_hidden', 32))
            self.z_select_tau         = float(kwargs.get('z_select_tau', 0.5))
            self.z_select_center_bias = float(kwargs.get('z_select_center_bias', 1.0))
            self.z_select_use_edges   = bool(kwargs.get('z_select_use_edges', True))
            self.z_select_norm_input  = bool(kwargs.get('z_select_norm_input', True))
            self.z_select_alpha_init  = float(kwargs.get('z_select_alpha_init', 0.1))

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

        # 2.5D: compress/select ref stack [B, K, H, W] -> [B, 1, H, W] per style stream
        if self.use_25d_style:
            valid_modes = ['conv', 'sobel', 'learned', 'learned_residual']
            if self.z_select_mode not in valid_modes:
                raise ValueError(f"Unknown z_select_mode: {self.z_select_mode}. Choose from {valid_modes}")

            if self.z_select_mode == 'conv':
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

            elif self.z_select_mode in ['learned', 'learned_residual']:
                K = self.ref_stack_size
                in_ch = 1 + K  # source_down + ref_stack_down
                if self.z_select_use_edges:
                    in_ch += 1 + K  # edge_source + edge_ref_stack
                self.z_selectors = nn.ModuleList([
                    LowResSliceSelector(in_ch=in_ch, ref_stack_size=K, hidden=self.z_select_hidden)
                    for _ in range(self.num_style_streams)
                ])
                if self.z_select_mode == 'learned_residual':
                    alpha = min(max(self.z_select_alpha_init, 1e-4), 1.0 - 1e-4)
                    alpha_logit = math.log(alpha / (1.0 - alpha))
                    self.z_select_alpha_logit = nn.Parameter(torch.tensor(alpha_logit, dtype=torch.float32))
            # sobel: no learnable params

        self.store_zselect_debug = True
        self._last_zselect_stats = {}

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
                                                 activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        self.conv11 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=True, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        self.conv12 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        self.conv21 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=True, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        self.conv22 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        self.conv31 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        self.conv32 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        self.conv41 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3, #feat_ch *4는 변치않게
                                upsample=True, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        self.conv42 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                upsample=False, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        self.conv51 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                upsample=True, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        self.conv52 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                upsample=False, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, )
        self.conv6 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, ) # 이걸 빠트렸었어.
                                # activate=False, demodulate=self.demodulate)

        # Separate style layers: 채널을 완전히 분리해서 각각 독립적으로 style 적용
        if self.use_separate_style_layers and self.use_triple_outputs:
            # 각 branch는 feat_ch 채널 (전체의 1/3)
            self.conv7_1 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d, )
            self.conv7_2 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d, )
            self.conv7_3 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d, )
            self.conv8_1 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d, )
            self.conv8_2 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d, )
            self.conv8_3 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d, )
            # 각각 독립적인 conv_final
            self.conv_final_1 = Conv(self.feat_ch, self.output_nc // 3, kernel_size=3, padding=1)
            self.conv_final_2 = Conv(self.feat_ch, self.output_nc // 3, kernel_size=3, padding=1)
            self.conv_final_3 = Conv(self.feat_ch, self.output_nc // 3, kernel_size=3, padding=1)
        elif self.use_separate_style_layers and self.use_multiple_outputs:
            # 각 branch는 feat_ch 채널 (전체의 절반)
            self.conv7_1 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d, )
            self.conv7_2 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d, )
            self.conv8_1 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d, )
            self.conv8_2 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d, )
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
        """2.5D conv mode: [B, num_streams*K, H, W] -> [B, num_streams, H, W]"""
        if not self.use_25d_style:
            return ref_all
        chunks = torch.split(ref_all, self.ref_stack_size, dim=1)
        style_maps = [z_agg(chunk) for chunk, z_agg in zip(chunks, self.z_aggs)]
        return torch.cat(style_maps, dim=1)

    def _style_size(self, source):
        H, W = source.shape[-2:]
        s = max(1, self.z_select_scale)
        return max(1, H // s), max(1, W // s)

    def _center_bias(self, logits):
        K = logits.shape[1]
        bias = torch.zeros_like(logits)
        bias[:, K // 2] = self.z_select_center_bias
        return bias

    def _store_zselect_stats(self, stream_idx, weights, selected, center, weighted, logits=None):
        if not getattr(self, 'store_zselect_debug', True):
            return
        with torch.no_grad():
            eps = 1e-8
            K = weights.shape[1]
            p = f"stream{stream_idx}"
            entropy = -(weights * (weights + eps).log()).sum(dim=1).mean()
            self._last_zselect_stats.update({
                f"{p}/entropy":              (entropy / math.log(K)).detach(),
                f"{p}/center_weight":         weights[:, K//2:K//2+1].mean().detach(),
                f"{p}/neighbor_mass":         (1.0 - weights[:, K//2:K//2+1].mean()).detach(),
                f"{p}/selected_center_delta": ((selected - center).abs().mean() / (center.abs().mean() + eps)).detach(),
                f"{p}/weighted_center_delta": ((weighted - center).abs().mean() / (center.abs().mean() + eps)).detach(),
                f"{p}/selected_std":          selected.std().detach(),
            })
            for i in range(K):
                self._last_zselect_stats[f"{p}/slice{i}_weight"] = weights[:, i].mean().detach()
            if self.z_select_mode == 'learned_residual':
                self._last_zselect_stats[f"{p}/alpha"] = torch.sigmoid(self.z_select_alpha_logit).detach()
            if logits is not None:
                self._last_zselect_stats[f"{p}/logits_std"] = logits.std().detach()

    def _select_one_ref_stream(self, source, ref_chunk, stream_idx=0, store_debug=True):
        K = ref_chunk.shape[1]
        center_idx = K // 2
        h, w = self._style_size(source)
        src_d = F.interpolate(source, size=(h, w), mode='bilinear', align_corners=False)
        ref_d = F.interpolate(ref_chunk, size=(h, w), mode='bilinear', align_corners=False)
        center = ref_d[:, center_idx:center_idx+1]

        src_in = spatial_zscore(src_d) if self.z_select_norm_input else src_d
        ref_in = spatial_zscore(ref_d) if self.z_select_norm_input else ref_d

        if self.z_select_mode == 'sobel':
            src_e = spatial_zscore(sobel_mag(src_d))
            ref_e = spatial_zscore(sobel_mag(ref_d))
            logits = -(ref_e - src_e).abs()
        elif self.z_select_mode in ['learned', 'learned_residual']:
            inputs = [src_in, ref_in]
            if self.z_select_use_edges:
                inputs += [spatial_zscore(sobel_mag(src_d)), spatial_zscore(sobel_mag(ref_d))]
            logits = self.z_selectors[stream_idx](torch.cat(inputs, dim=1))
        else:
            raise RuntimeError(f"Invalid mode for _select_one_ref_stream: {self.z_select_mode}")

        logits_biased = logits + self._center_bias(logits)
        weights = torch.softmax(logits_biased / max(self.z_select_tau, 1e-6), dim=1)
        weighted = (weights * ref_d).sum(dim=1, keepdim=True)

        if self.z_select_mode == 'learned_residual':
            alpha = torch.sigmoid(self.z_select_alpha_logit)
            selected = center + alpha * (weighted - center)
        else:
            selected = weighted

        if store_debug:
            self._store_zselect_stats(stream_idx, weights, selected, center, weighted, logits)

        return selected

    def _make_style_guidance(self, source, ref_all, encode_only=False):
        """Build style_guidance_1 from source + ref_all."""
        if not self.use_25d_style:
            h, w = self._style_size(source)
            return F.interpolate(ref_all, size=(h, w), mode='nearest')

        if self.z_select_mode == 'conv':
            ref = self._aggregate_ref_stack(ref_all)
            if self.is_3d:
                ref = ref.permute(0, 1, 4, 2, 3)
                sg = F.interpolate(ref, scale_factor=1/self.z_select_scale, mode='trilinear', align_corners=False)
                return sg.permute(0, 1, 3, 4, 2)
            h, w = self._style_size(source)
            return F.interpolate(ref, size=(h, w), mode='nearest')

        # sobel / learned / learned_residual
        chunks = torch.split(ref_all, self.ref_stack_size, dim=1)
        store = (not encode_only) and getattr(self, 'store_zselect_debug', True)
        if store:
            self._last_zselect_stats = {}
        selected_maps = [
            self._select_one_ref_stream(source, chunk, stream_idx=i, store_debug=store)
            for i, chunk in enumerate(chunks)
        ]
        return torch.cat(selected_maps, dim=1)

    def forward(self, merged_input, layers=[], encode_only=False):

        x = merged_input[:, :1, ...]
        ref_all = merged_input[:, 1:, ...]
        style_guidance_1 = self._make_style_guidance(x, ref_all, encode_only=encode_only)

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
        if self.use_separate_style_layers and self.use_triple_outputs:
            # feat6를 3등분으로 분리
            feat6_1, feat6_2, feat6_3 = torch.chunk(feat6, chunks=3, dim=1)  # 각 [B, feat_ch, H, W]
            # style도 분리
            style_1 = style_guidance_1[:, :1, ...]  # [B, 1, ...]
            style_2 = style_guidance_1[:, 1:2, ...]  # [B, 1, ...]
            style_3 = style_guidance_1[:, 2:3, ...]  # [B, 1, ...]

            # conv7: 각각 독립적으로 처리
            feat7_1 = self.conv7_1(feat6_1, style_1)  # [B, feat_ch, H, W]
            feat7_2 = self.conv7_2(feat6_2, style_2)  # [B, feat_ch, H, W]
            feat7_3 = self.conv7_3(feat6_3, style_3)  # [B, feat_ch, H, W]

            # conv8: 각각 독립적으로 처리
            feat8_1 = self.conv8_1(feat7_1, style_1)  # [B, feat_ch, H, W]
            feat8_2 = self.conv8_2(feat7_2, style_2)  # [B, feat_ch, H, W]
            feat8_3 = self.conv8_3(feat7_3, style_3)  # [B, feat_ch, H, W]

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
                 noise_independent=False):

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

    def forward(self, x, style):

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
            if style.shape[1] == 1:
                x = self.normalize(x)
            elif style.shape[1] == 2:
                x1, x2 = torch.chunk(x, chunks=2, dim=1)  # 각 부분 [B, C/2, H, W]
                x1 = self.normalize(x1)
                x2 = self.normalize(x2)
                x = torch.cat((x1, x2), dim=1)
            elif style.shape[1] == 3:
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
            if style.shape[1] == 1:
                x = self.normalize(x)
                if self.randomize_noise:
                    noise = torch.randn_like(x) * self.noise_strength
                else:
                    noise = torch.zeros_like(x) * self.noise_strength
                x = x + noise
            elif style.shape[1] == 2:
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
            elif style.shape[1] == 3:
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
            elif style.shape[1] == 3:
                actv1 = self.mlp_shared(style[:, :1, ...])
                gamma = self.mlp_gamma(actv1)  # B, feat_ch//3, f_H, f_W
                beta = self.mlp_beta(actv1)
                actv_2 = self.mlp_shared_2(style[:, 1:2, ...])
                gamma_2 = self.mlp_gamma_2(actv_2)
                beta_2 = self.mlp_beta_2(actv_2)
                actv_3 = self.mlp_shared_3(style[:, 2:3, ...])
                gamma_3 = self.mlp_gamma_3(actv_3)
                beta_3 = self.mlp_beta_3(actv_3)
                gamma = torch.cat((gamma, gamma_2, gamma_3), dim=1)  # B, feat_ch, f_H, f_W
                beta = torch.cat((beta, beta_2, beta_3), dim=1)

            # 3. 거기서 감마 베타 나눠서 denorm
            x_pre = x
            x = x * gamma + beta
            if getattr(self, '_log_style_modulation', False):
                with torch.no_grad():
                    eps = 1e-8
                    self.last_style_debug = {
                        'gamma_abs':         gamma.abs().mean().detach(),
                        'beta_abs':          beta.abs().mean().detach(),
                        'gamma_spatial_std': gamma.std(dim=(-2,-1)).mean().detach(),
                        'beta_spatial_std':  beta.std(dim=(-2,-1)).mean().detach(),
                        'delta_ratio':       ((x - x_pre).abs().mean() / (x_pre.abs().mean() + eps)).detach(),
                        'x_out_over_x_pre':  (x.abs().mean() / (x_pre.abs().mean() + eps)).detach(),
                    }
        
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