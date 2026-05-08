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
            # Confidence gating (opt-in). When True, build a confidence branch
            # that gates StyleConv modulation residually. Disabled in 3D path.
            self.use_confidence_gating = kwargs.get('use_confidence_gating', False)
            self.conf_init_bias = kwargs.get('conf_init_bias', 2.0)
            self.conf_detach = kwargs.get('conf_detach', False)

        except KeyError as e:
            raise ValueError(f"Missing required parameter: {str(e)}")

        # Effective gating flag: only active in 2D / 2.5D paths.
        self._gating_active = bool(self.use_confidence_gating) and (not self.is_3d)
        self.last_conf_map = None

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

        # Confidence head(s). Built only when gating is active.
        # 2.5D: per-stream head consumes K-channel ref stack -> 1ch logit.
        # 2D  : single head consumes num_streams-ch ref -> num_streams-ch logits.
        if self._gating_active:
            conf_hidden = max(8, self.feat_ch // 8)
            conf_init_bias = self.conf_init_bias

            def _make_conf_head(in_ch, out_ch):
                head = nn.Sequential(
                    nn.Conv2d(in_ch, conf_hidden, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(conf_hidden, out_ch, kernel_size=3, padding=1),
                )
                # Init: keep conf high at start so behavior is close to baseline.
                nn.init.zeros_(head[-1].weight)
                nn.init.constant_(head[-1].bias, conf_init_bias)
                return head

            if self.use_25d_style:
                self.q_aggs = nn.ModuleList([
                    _make_conf_head(self.ref_stack_size, 1)
                    for _ in range(self.num_style_streams)
                ])
            else:
                self.q_net = _make_conf_head(self.num_style_streams, self.num_style_streams)

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
                                                 activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, use_confidence_gating=self._gating_active)
        self.conv11 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=True, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, use_confidence_gating=self._gating_active)
        self.conv12 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, use_confidence_gating=self._gating_active)
        self.conv21 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=True, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, use_confidence_gating=self._gating_active)
        self.conv22 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, use_confidence_gating=self._gating_active)
        self.conv31 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, use_confidence_gating=self._gating_active)
        self.conv32 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, use_confidence_gating=self._gating_active)
        self.conv41 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3, #feat_ch *4는 변치않게
                                upsample=True, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, use_confidence_gating=self._gating_active)
        self.conv42 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                upsample=False, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, use_confidence_gating=self._gating_active)
        self.conv51 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                upsample=True, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, use_confidence_gating=self._gating_active)
        self.conv52 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                upsample=False, activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, use_confidence_gating=self._gating_active)
        self.conv6 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                activate=True, demodulate=self.demodulate, ch=ch, is_3d=self.is_3d, noise_independent=self.noise_independent, use_confidence_gating=self._gating_active) # 이걸 빠트렸었어.
                                # activate=False, demodulate=self.demodulate)

        # Separate style layers: 채널을 완전히 분리해서 각각 독립적으로 style 적용
        if self.use_separate_style_layers and self.use_triple_outputs:
            # 각 branch는 feat_ch 채널 (전체의 1/3)
            self.conv7_1 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d, use_confidence_gating=self._gating_active)
            self.conv7_2 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d, use_confidence_gating=self._gating_active)
            self.conv7_3 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d, use_confidence_gating=self._gating_active)
            self.conv8_1 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d, use_confidence_gating=self._gating_active)
            self.conv8_2 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d, use_confidence_gating=self._gating_active)
            self.conv8_3 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d, use_confidence_gating=self._gating_active)
            # 각각 독립적인 conv_final
            self.conv_final_1 = Conv(self.feat_ch, self.output_nc // 3, kernel_size=3, padding=1)
            self.conv_final_2 = Conv(self.feat_ch, self.output_nc // 3, kernel_size=3, padding=1)
            self.conv_final_3 = Conv(self.feat_ch, self.output_nc // 3, kernel_size=3, padding=1)
        elif self.use_separate_style_layers and self.use_multiple_outputs:
            # 각 branch는 feat_ch 채널 (전체의 절반)
            self.conv7_1 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d, use_confidence_gating=self._gating_active)
            self.conv7_2 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d, use_confidence_gating=self._gating_active)
            self.conv8_1 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d, use_confidence_gating=self._gating_active)
            self.conv8_2 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                     activate=True, demodulate=self.demodulate, ch=1, is_3d=self.is_3d, use_confidence_gating=self._gating_active)
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
        chunks = torch.split(ref_all, self.ref_stack_size, dim=1)
        style_maps = [z_agg(chunk) for chunk, z_agg in zip(chunks, self.z_aggs)]
        return torch.cat(style_maps, dim=1)

    def _build_style_and_conf(self, ref_all):
        """Returns (style, conf). conf is None when gating is inactive.
        2D : ref_all [B, num_streams, H, W] -> conf [B, num_streams, H, W]
        2.5D: ref_all [B, num_streams*K, H, W] -> conf [B, num_streams, H, W]
        """
        style = self._aggregate_ref_stack(ref_all)
        if not self._gating_active:
            return style, None
        if self.use_25d_style:
            chunks = torch.split(ref_all, self.ref_stack_size, dim=1)
            conf_maps = [torch.sigmoid(q_agg(chunk))
                         for chunk, q_agg in zip(chunks, self.q_aggs)]
            conf = torch.cat(conf_maps, dim=1)
        else:
            conf = torch.sigmoid(self.q_net(ref_all))
        return style, conf

    def forward(self, merged_input, layers=[], encode_only=False):

        x = merged_input[:, :1, ...]
        ref_all = merged_input[:, 1:, ...]
        ref, conf = self._build_style_and_conf(ref_all)  # conf is None when gating off

        if self.is_3d:
            ref = ref.permute(0, 1, 4, 2, 3)
            style_guidance_1 = F.interpolate(ref, scale_factor=1/16, mode='trilinear', align_corners=False)
            style_guidance_1 = style_guidance_1.permute(0, 1, 3, 4, 2)
        else:
            style_guidance_1 = F.interpolate(ref, scale_factor=1/16, mode='nearest') # Final #TODO: sliding infer로 줄어든만큼 이것도 줄여줘야해. 8정도가 적절할듯

        # Confidence guidance at the same low-res scale as style. Disabled in 3D.
        conf_guidance_1 = None
        if conf is not None:
            conf_guidance_1 = F.interpolate(conf, scale_factor=1/16, mode='nearest')
            if self.conf_detach:
                conf_guidance_1 = conf_guidance_1.detach()
        # Cache for module-level regularization (overwritten each forward).
        self.last_conf_map = conf
        
        # style_guidance_1 = F.interpolate(ref, scale_factor=1/8, mode='bilinear', align_corners=True) # Ablation
        # style_guidance_1 = F.interpolate(ref, scale_factor=1/32, mode='bilinear', align_corners=True) # Ablation
        # style_guidance_1 = F.interpolate(ref, scale_factor=1/64, mode='bilinear', align_corners=True) # Ablation
        # style_guidance_1 = F.interpolate(ref, size=(1, 1), mode='nearest') # Ablation
        feats = []
        feat0 = self.conv0(x, style_guidance_1, conf_guidance_1) # [1, feat_ch, H, W]
        feat1 = self.conv11(feat0, style_guidance_1, conf_guidance_1) # [1, feat_ch, H/2, W/2]
        feat1 = self.conv12(feat1, style_guidance_1, conf_guidance_1) # [1, feat_ch, H/2, W/2]
        feat2 = self.conv21(feat1, style_guidance_1, conf_guidance_1) # [1, feat_ch, H/4, W/4]
        feat2 = self.conv22(feat2, style_guidance_1, conf_guidance_1) # [1, feat_ch, H/4, W/4]
        feat3 = self.conv31(feat2, style_guidance_1, conf_guidance_1) # [1, feat_ch, H/4, W/4] #TODO: 메모리 많아서 뺌.
        feat3 = self.conv32(feat3, style_guidance_1, conf_guidance_1) # [1, feat_ch, H/4, W/4]
        feat4 = self.conv41(feat3 + feat2, style_guidance_1, conf_guidance_1)# [1, feat_ch, H/2, W/2] #TODO: 원래 intput feat3 + feat2
        feat4 = self.conv42(feat4, style_guidance_1, conf_guidance_1)        # [1, feat_ch, H/2, W/2]
        feat5 = self.conv51(feat4 + feat1, style_guidance_1, conf_guidance_1)# [1, feat_ch, H, W]
        feat5 = self.conv52(feat5, style_guidance_1, conf_guidance_1)        # [1, feat_ch, H, W]
        feat6 = self.conv6(feat5 + feat0, style_guidance_1, conf_guidance_1) # [1, feat_ch, H, W]

        # Separate style layers: 채널을 완전히 분리해서 각각 독립적으로 처리
        if self.use_separate_style_layers and self.use_triple_outputs:
            # feat6를 3등분으로 분리
            feat6_1, feat6_2, feat6_3 = torch.chunk(feat6, chunks=3, dim=1)  # 각 [B, feat_ch, H, W]
            # style도 분리
            style_1 = style_guidance_1[:, :1, ...]  # [B, 1, ...]
            style_2 = style_guidance_1[:, 1:2, ...]  # [B, 1, ...]
            style_3 = style_guidance_1[:, 2:3, ...]  # [B, 1, ...]
            # conf도 같은 방식으로 분리 (None일 땐 그대로 None)
            if conf_guidance_1 is not None:
                conf_1 = conf_guidance_1[:, :1, ...]
                conf_2 = conf_guidance_1[:, 1:2, ...]
                conf_3 = conf_guidance_1[:, 2:3, ...]
            else:
                conf_1 = conf_2 = conf_3 = None

            # conv7: 각각 독립적으로 처리
            feat7_1 = self.conv7_1(feat6_1, style_1, conf_1)  # [B, feat_ch, H, W]
            feat7_2 = self.conv7_2(feat6_2, style_2, conf_2)  # [B, feat_ch, H, W]
            feat7_3 = self.conv7_3(feat6_3, style_3, conf_3)  # [B, feat_ch, H, W]

            # conv8: 각각 독립적으로 처리
            feat8_1 = self.conv8_1(feat7_1, style_1, conf_1)  # [B, feat_ch, H, W]
            feat8_2 = self.conv8_2(feat7_2, style_2, conf_2)  # [B, feat_ch, H, W]
            feat8_3 = self.conv8_3(feat7_3, style_3, conf_3)  # [B, feat_ch, H, W]

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
            # conf도 같은 방식으로 분리 (None일 땐 그대로 None)
            if conf_guidance_1 is not None:
                conf_1 = conf_guidance_1[:, :1, ...]
                conf_2 = conf_guidance_1[:, 1:, ...]
            else:
                conf_1 = conf_2 = None

            # conv7: 각각 독립적으로 처리
            feat7_1 = self.conv7_1(feat6_1, style_1, conf_1)  # [B, feat_ch, H, W]
            feat7_2 = self.conv7_2(feat6_2, style_2, conf_2)  # [B, feat_ch, H, W]

            # conv8: 각각 독립적으로 처리
            feat8_1 = self.conv8_1(feat7_1, style_1, conf_1)  # [B, feat_ch, H, W]
            feat8_2 = self.conv8_2(feat7_2, style_2, conf_2)  # [B, feat_ch, H, W]

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


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConstantChannelEncoder(nn.Module):
    def __init__(self, in_ch=1, feat_ch=128):
        super().__init__()
        self.stem = ConvBlock(in_ch, feat_ch)

        self.down1 = self._make_down(feat_ch)
        self.down2 = self._make_down(feat_ch)
        self.down3 = self._make_down(feat_ch)
        self.down4 = self._make_down(feat_ch)

    @staticmethod
    def _make_down(feat_ch):
        return nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(feat_ch, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            ConvBlock(feat_ch, feat_ch),
        )

    def forward(self, x):
        f1 = self.stem(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        f5 = self.down4(f4)
        return [f1, f2, f3, f4, f5]


class Local2p5DCrossAttention(nn.Module):
    def __init__(
        self,
        feat_ch=128,
        attn_ch=64,
        ref_stack_size=3,
        window_size=3,
        temperature=1.0,
        center_bias_z=0.05,
        center_bias_xy=0.02,
    ):
        super().__init__()
        if window_size % 2 != 1:
            raise ValueError("window_size must be odd.")

        self.feat_ch = feat_ch
        self.attn_ch = attn_ch
        self.K = ref_stack_size
        self.window_size = window_size
        self.radius = window_size // 2
        self.R2 = window_size * window_size
        self.temperature = temperature
        self.center_bias_z = center_bias_z
        self.center_bias_xy = center_bias_xy

        self.q_proj = nn.Conv2d(feat_ch, attn_ch, kernel_size=1)
        self.k_proj = nn.Conv2d(feat_ch, attn_ch, kernel_size=1)
        self.v_proj = nn.Conv2d(feat_ch, feat_ch, kernel_size=1)
        self.out_proj = nn.Conv2d(feat_ch, feat_ch, kernel_size=1)

        z_offsets = torch.arange(ref_stack_size, dtype=torch.float32) - (ref_stack_size // 2)
        self.register_buffer("z_offsets", z_offsets)

        dx_list, dy_list = [], []
        for dy in range(-self.radius, self.radius + 1):
            for dx in range(-self.radius, self.radius + 1):
                dy_list.append(dy)
                dx_list.append(dx)
        self.register_buffer("dx_offsets", torch.tensor(dx_list, dtype=torch.float32))
        self.register_buffer("dy_offsets", torch.tensor(dy_list, dtype=torch.float32))

    def _unfold(self, x, channels, h, w):
        patches = F.unfold(
            x,
            kernel_size=self.window_size,
            padding=self.radius,
        )
        return patches.view(x.shape[0], channels, self.R2, h, w)

    def forward(self, src_feat, ref_feat):
        B, C, h, w = src_feat.shape
        B2, K, C2, h2, w2 = ref_feat.shape
        if B != B2 or C != C2 or K != self.K or h != h2 or w != w2:
            raise ValueError(
                "Expected src_feat [B,C,h,w] and ref_feat [B,K,C,h,w] "
                f"with K={self.K}, got {tuple(src_feat.shape)} and {tuple(ref_feat.shape)}."
            )

        q = self.q_proj(src_feat)
        q = F.normalize(q, dim=1)
        q = q[:, None, :, None, :, :]

        ref_flat = ref_feat.reshape(B * K, C, h, w)
        k_flat = self.k_proj(ref_flat)
        v_flat = self.v_proj(ref_flat)

        k_patches = self._unfold(k_flat, self.attn_ch, h, w)
        v_patches = self._unfold(v_flat, C, h, w)

        k_patches = k_patches.view(B, K, self.attn_ch, self.R2, h, w)
        v_patches = v_patches.view(B, K, C, self.R2, h, w)
        k_patches = F.normalize(k_patches, dim=2)

        scores = (q * k_patches).sum(dim=2)
        z_penalty = self.center_bias_z * self.z_offsets.abs().view(1, K, 1, 1, 1)
        xy_penalty = self.center_bias_xy * (
            self.dx_offsets.abs() + self.dy_offsets.abs()
        ).view(1, 1, self.R2, 1, 1)
        scores = scores - z_penalty - xy_penalty

        scores_flat = scores.reshape(B, K * self.R2, h, w)
        weights_flat = torch.softmax(
            scores_flat / max(float(self.temperature), 1e-6),
            dim=1,
        )
        weights = weights_flat.view(B, K, self.R2, h, w)

        attn_feat = (weights[:, :, None, :, :, :] * v_patches).sum(dim=(1, 3))
        attn_feat = self.out_proj(attn_feat)

        z_weights = weights.sum(dim=2)
        exp_dz = (
            weights * self.z_offsets.view(1, K, 1, 1, 1)
        ).sum(dim=(1, 2)).unsqueeze(1)
        exp_dx = (
            weights * self.dx_offsets.view(1, 1, self.R2, 1, 1)
        ).sum(dim=(1, 2)).unsqueeze(1)
        exp_dy = (
            weights * self.dy_offsets.view(1, 1, self.R2, 1, 1)
        ).sum(dim=(1, 2)).unsqueeze(1)

        aux = {
            "weights": weights,
            "z_weights": z_weights,
            "exp_dz": exp_dz,
            "exp_dx": exp_dx,
            "exp_dy": exp_dy,
        }
        return attn_feat, aux


class MultiHeadLocal2p5DCrossAttention(nn.Module):
    """Multi-head local 2.5D cross-attention over K slices and a local xy window."""

    def __init__(
        self,
        feat_ch=192,
        nhead=4,
        ref_stack_size=5,
        window_size=5,
        temperature=1.0,
        center_bias_z=0.05,
        center_bias_xy=0.02,
        use_relative_bias=True,
    ):
        super().__init__()
        if window_size % 2 != 1:
            raise ValueError("window_size must be odd.")
        if feat_ch % nhead != 0:
            raise ValueError(f"feat_ch={feat_ch} must be divisible by nhead={nhead}.")

        self.feat_ch = feat_ch
        self.nhead = nhead
        self.head_dim = feat_ch // nhead
        self.K = ref_stack_size
        self.window_size = window_size
        self.radius = window_size // 2
        self.R2 = window_size * window_size
        self.temperature = temperature
        self.center_bias_z = center_bias_z
        self.center_bias_xy = center_bias_xy
        self.use_relative_bias = use_relative_bias

        self.q_proj = nn.Conv2d(feat_ch, feat_ch, kernel_size=1)
        self.k_proj = nn.Conv2d(feat_ch, feat_ch, kernel_size=1)
        self.v_proj = nn.Conv2d(feat_ch, feat_ch, kernel_size=1)
        self.out_proj = nn.Conv2d(feat_ch, feat_ch, kernel_size=1)

        z_offsets = torch.arange(ref_stack_size, dtype=torch.float32) - (ref_stack_size // 2)
        self.register_buffer("z_offsets", z_offsets)

        dx_list, dy_list = [], []
        for dy in range(-self.radius, self.radius + 1):
            for dx in range(-self.radius, self.radius + 1):
                dy_list.append(dy)
                dx_list.append(dx)
        self.register_buffer("dx_offsets", torch.tensor(dx_list, dtype=torch.float32))
        self.register_buffer("dy_offsets", torch.tensor(dy_list, dtype=torch.float32))

        if use_relative_bias:
            self.rel_bias = nn.Parameter(torch.zeros(nhead, ref_stack_size, self.R2))
        else:
            self.rel_bias = None

        self.scale = self.head_dim ** -0.5

    def _unfold_heads(self, x, h, w):
        BK, C, _, _ = x.shape
        patches = F.unfold(
            x,
            kernel_size=self.window_size,
            padding=self.radius,
        )
        return patches.view(BK, self.nhead, self.head_dim, self.R2, h, w)

    def forward(self, src_feat, ref_feat):
        B, C, h, w = src_feat.shape
        B2, K, C2, h2, w2 = ref_feat.shape

        if B != B2 or C != C2 or K != self.K or h != h2 or w != w2:
            raise ValueError(
                "Expected src_feat [B,C,h,w] and ref_feat [B,K,C,h,w], "
                f"got {tuple(src_feat.shape)} and {tuple(ref_feat.shape)}."
            )

        q = self.q_proj(src_feat)
        q = q.view(B, self.nhead, self.head_dim, h, w)
        q = F.normalize(q, dim=2)

        ref_flat = ref_feat.reshape(B * K, C, h, w)
        k_flat = self.k_proj(ref_flat)
        v_flat = self.v_proj(ref_flat)

        k_patches = self._unfold_heads(k_flat, h, w)
        v_patches = self._unfold_heads(v_flat, h, w)

        k_patches = k_patches.view(B, K, self.nhead, self.head_dim, self.R2, h, w)
        v_patches = v_patches.view(B, K, self.nhead, self.head_dim, self.R2, h, w)
        k_patches = k_patches.permute(0, 2, 1, 4, 3, 5, 6)
        v_patches = v_patches.permute(0, 2, 1, 4, 3, 5, 6)
        k_patches = F.normalize(k_patches, dim=4)

        q = q[:, :, None, None, :, :, :]
        scores = (q * k_patches).sum(dim=4) * self.scale

        z_penalty = self.center_bias_z * self.z_offsets.abs().view(1, 1, K, 1, 1, 1)
        xy_penalty = self.center_bias_xy * (
            self.dx_offsets.abs() + self.dy_offsets.abs()
        ).view(1, 1, 1, self.R2, 1, 1)
        scores = scores - z_penalty - xy_penalty

        if self.rel_bias is not None:
            scores = scores + self.rel_bias.view(1, self.nhead, K, self.R2, 1, 1)

        scores_flat = scores.reshape(B, self.nhead, K * self.R2, h, w)
        weights_flat = torch.softmax(
            scores_flat / max(float(self.temperature), 1.0e-6),
            dim=2,
        )
        weights = weights_flat.view(B, self.nhead, K, self.R2, h, w)

        out = (weights[:, :, :, :, None, :, :] * v_patches).sum(dim=(2, 3))
        out = out.reshape(B, C, h, w)
        out = self.out_proj(out)

        weights_mean = weights.mean(dim=1)
        z_weights = weights_mean.sum(dim=2)
        exp_dz = (
            weights_mean * self.z_offsets.view(1, K, 1, 1, 1)
        ).sum(dim=(1, 2)).unsqueeze(1)
        exp_dx = (
            weights_mean * self.dx_offsets.view(1, 1, self.R2, 1, 1)
        ).sum(dim=(1, 2)).unsqueeze(1)
        exp_dy = (
            weights_mean * self.dy_offsets.view(1, 1, self.R2, 1, 1)
        ).sum(dim=(1, 2)).unsqueeze(1)
        weights_safe = weights_flat.clamp_min(1.0e-8)
        entropy = -(weights_safe * weights_safe.log()).sum(dim=2).mean()

        aux = {
            "weights": weights_mean,
            "z_weights": z_weights,
            "exp_dz": exp_dz,
            "exp_dx": exp_dx,
            "exp_dy": exp_dy,
            "entropy": entropy.detach(),
        }
        return out, aux


class Local2p5DTransformerBlock(nn.Module):
    """Source-dominant local 2.5D cross-attention block with gated residual and FFN."""

    def __init__(
        self,
        feat_ch=192,
        nhead=4,
        ref_stack_size=5,
        window_size=5,
        temperature=1.0,
        center_bias_z=0.05,
        center_bias_xy=0.02,
        mlp_ratio=2.0,
        gate_init=-2.0,
        use_relative_bias=True,
    ):
        super().__init__()
        self.gate_init = gate_init

        self.norm_src = nn.GroupNorm(1, feat_ch)
        self.norm_ref = nn.GroupNorm(1, feat_ch)
        self.norm_ffn = nn.GroupNorm(1, feat_ch)

        self.attn = MultiHeadLocal2p5DCrossAttention(
            feat_ch=feat_ch,
            nhead=nhead,
            ref_stack_size=ref_stack_size,
            window_size=window_size,
            temperature=temperature,
            center_bias_z=center_bias_z,
            center_bias_xy=center_bias_xy,
            use_relative_bias=use_relative_bias,
        )

        self.gate_attn = nn.Conv2d(feat_ch * 2, feat_ch, kernel_size=3, padding=1)
        hidden = int(feat_ch * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(feat_ch, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, feat_ch, kernel_size=1),
        )
        self.gate_ffn = nn.Parameter(torch.tensor(float(gate_init)))

        self.last_gate_attn_mean = None
        self.last_gate_ffn = None
        self.reset_gate()

    def reset_gate(self):
        nn.init.zeros_(self.gate_attn.weight)
        nn.init.constant_(self.gate_attn.bias, self.gate_init)
        with torch.no_grad():
            self.gate_ffn.fill_(float(self.gate_init))

    def _norm_ref_stack(self, ref_feat):
        B, K, C, H, W = ref_feat.shape
        ref_flat = ref_feat.reshape(B * K, C, H, W)
        ref_flat = self.norm_ref(ref_flat)
        return ref_flat.view(B, K, C, H, W)

    def forward(self, src_feat, ref_feat):
        src_norm = self.norm_src(src_feat)
        ref_norm = self._norm_ref_stack(ref_feat)

        attn_out, aux = self.attn(src_norm, ref_norm)

        gate = torch.sigmoid(self.gate_attn(torch.cat([src_feat, attn_out], dim=1)))
        self.last_gate_attn_mean = gate.detach().mean()
        h = src_feat + gate * attn_out

        ffn_out = self.ffn(self.norm_ffn(h))
        gate_ffn = torch.sigmoid(self.gate_ffn)
        self.last_gate_ffn = gate_ffn.detach()
        h = h + gate_ffn * ffn_out

        aux["gate_attn"] = self.last_gate_attn_mean
        aux["gate_ffn"] = self.last_gate_ffn
        return h, aux


class CrossAttnFuseBlock(nn.Module):
    def __init__(self, feat_ch=128):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(feat_ch * 2, feat_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(feat_ch, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(feat_ch, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, src_feat, attn_ref_feat):
        return self.fuse(torch.cat([src_feat, attn_ref_feat], dim=1))


class UpBlock(nn.Module):
    def __init__(self, feat_ch=128):
        super().__init__()
        self.conv = ConvBlock(feat_ch * 2, feat_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ProposedSynthesisCrossAttn(nn.Module):
    """StyleConv-free reference-guided synthesis network with local 2.5D cross-attention."""

    def __init__(
        self,
        input_nc=1,
        output_nc=1,
        feat_ch=128,
        ref_stack_size=3,
        cross_attn_ch=64,
        cross_attn_window=3,
        cross_attn_temperature=1.0,
        cross_attn_center_bias_z=0.05,
        cross_attn_center_bias_xy=0.02,
        residual_fusion=True,
        fusion_gate_init=-2.0,
        use_transformer_attn=False,
        cross_attn_nhead=4,
        cross_attn_mlp_ratio=2.0,
        cross_attn_relative_bias=True,
        cross_attn_gate_init=-2.0,
        use_multiple_outputs=False,
        use_triple_outputs=False,
        is_3d=False,
        **kwargs,
    ):
        super().__init__()
        if is_3d:
            raise NotImplementedError("ProposedSynthesisCrossAttn is 2D/2.5D only.")
        if use_multiple_outputs or use_triple_outputs:
            raise NotImplementedError("ProposedSynthesisCrossAttn supports single-output only.")

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.feat_ch = feat_ch
        self.ref_stack_size = ref_stack_size
        self.residual_fusion = residual_fusion
        self.fusion_gate_init = fusion_gate_init
        self.use_transformer_attn = use_transformer_attn
        self.cross_attn_nhead = cross_attn_nhead
        self.cross_attn_mlp_ratio = cross_attn_mlp_ratio
        self.cross_attn_relative_bias = cross_attn_relative_bias
        self.cross_attn_gate_init = cross_attn_gate_init
        self.last_attn_aux = None
        self.last_conf_map = None

        self.source_encoder = ConstantChannelEncoder(in_ch=input_nc, feat_ch=feat_ch)
        self.ref_encoder = ConstantChannelEncoder(in_ch=1, feat_ch=feat_ch)

        if self.use_transformer_attn:
            trans_kwargs = dict(
                feat_ch=feat_ch,
                nhead=cross_attn_nhead,
                ref_stack_size=ref_stack_size,
                window_size=cross_attn_window,
                temperature=cross_attn_temperature,
                center_bias_z=cross_attn_center_bias_z,
                center_bias_xy=cross_attn_center_bias_xy,
                mlp_ratio=cross_attn_mlp_ratio,
                gate_init=cross_attn_gate_init,
                use_relative_bias=cross_attn_relative_bias,
            )
            self.trans3 = Local2p5DTransformerBlock(**trans_kwargs)
            self.trans4 = Local2p5DTransformerBlock(**trans_kwargs)
            self.trans5 = Local2p5DTransformerBlock(**trans_kwargs)
            self.attn3 = self.attn4 = self.attn5 = None
            self.fuse3 = self.fuse4 = self.fuse5 = None
        else:
            attn_kwargs = dict(
                feat_ch=feat_ch,
                attn_ch=cross_attn_ch,
                ref_stack_size=ref_stack_size,
                window_size=cross_attn_window,
                temperature=cross_attn_temperature,
                center_bias_z=cross_attn_center_bias_z,
                center_bias_xy=cross_attn_center_bias_xy,
            )
            self.attn3 = Local2p5DCrossAttention(**attn_kwargs)
            self.attn4 = Local2p5DCrossAttention(**attn_kwargs)
            self.attn5 = Local2p5DCrossAttention(**attn_kwargs)
            self.trans3 = self.trans4 = self.trans5 = None

            self.fuse3 = CrossAttnFuseBlock(feat_ch)
            self.fuse4 = CrossAttnFuseBlock(feat_ch)
            self.fuse5 = CrossAttnFuseBlock(feat_ch)

        self.up4 = UpBlock(feat_ch)
        self.up3 = UpBlock(feat_ch)
        self.up2 = UpBlock(feat_ch)
        self.up1 = UpBlock(feat_ch)

        self.final = nn.Conv2d(feat_ch, output_nc, kernel_size=3, padding=1)

    def reset_fusion_gates(self):
        for name in ("trans3", "trans4", "trans5", "fuse3", "fuse4", "fuse5"):
            module = getattr(self, name, None)
            if module is not None and hasattr(module, "reset_gate"):
                module.reset_gate()

    def _encode_ref_stack(self, ref_stack):
        B, K, H, W = ref_stack.shape
        ref_flat = ref_stack.reshape(B * K, 1, H, W)
        feats_flat = self.ref_encoder(ref_flat)

        feats = []
        for f in feats_flat:
            _, C, h, w = f.shape
            feats.append(f.view(B, K, C, h, w))
        return feats

    def forward(self, merged_input, layers=[], encode_only=False):
        x = merged_input[:, :self.input_nc, ...]
        ref_stack = merged_input[:, self.input_nc:, ...]

        if ref_stack.dim() != 4:
            raise ValueError(f"Expected 2D merged input [B,1+K,H,W], got {tuple(merged_input.shape)}.")
        B, K, H, W = ref_stack.shape
        if K != self.ref_stack_size:
            raise ValueError(f"Expected K={self.ref_stack_size}, got {K}.")

        Fs1, Fs2, Fs3, Fs4, Fs5 = self.source_encoder(x)
        Fr1, Fr2, Fr3, Fr4, Fr5 = self._encode_ref_stack(ref_stack)

        if self.use_transformer_attn:
            H3, aux3 = self.trans3(Fs3, Fr3)
            H4, aux4 = self.trans4(Fs4, Fr4)
            H5, aux5 = self.trans5(Fs5, Fr5)
        else:
            A3, aux3 = self.attn3(Fs3, Fr3)
            A4, aux4 = self.attn4(Fs4, Fr4)
            A5, aux5 = self.attn5(Fs5, Fr5)

            H3 = self.fuse3(Fs3, A3)
            H4 = self.fuse4(Fs4, A4)
            H5 = self.fuse5(Fs5, A5)

        self.last_attn_aux = {
            "s3": aux3,
            "s4": aux4,
            "s5": aux5,
        }

        D4 = self.up4(H5, H4)
        D3 = self.up3(D4, H3)
        D2 = self.up2(D3, Fs2)
        D1 = self.up1(D2, Fs1)

        out = torch.tanh(self.final(D1))

        if encode_only:
            feat_list = [Fs1, Fs2, H3, H4, H5, D3, D2]
            if layers is None or len(layers) == 0:
                return feat_list
            return [feat_list[i] for i in layers]

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
                 use_confidence_gating=False):

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
        # When True: gamma/beta MLP final layers are zero-initialized so that
        # initial modulation is near-identity (residual gated form expects this).
        self.use_confidence_gating = use_confidence_gating

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

        # Zero-init gamma/beta when gated so that modulation is identity at start.
        # In residual form: x*(1 + conf*tanh(gamma)) + conf*beta -> equals x when
        # gamma=beta=0 regardless of conf. Lets the network learn modulation gradually.
        if self.use_confidence_gating:
            for layer_name in ('mlp_gamma', 'mlp_beta',
                               'mlp_gamma_2', 'mlp_beta_2',
                               'mlp_gamma_3', 'mlp_beta_3'):
                if hasattr(self, layer_name):
                    layer = getattr(self, layer_name)
                    nn.init.zeros_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

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

    def forward(self, x, style, conf=None): # style: [1, 64, 1, 1]; conf: [B, num_streams, h, w] or None

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
            if conf is None:
                x = x * gamma + beta
            else:
                # Residual gated modulation:
                #   conf=0 -> x unchanged (untrusted ref ignored)
                #   conf=1 -> full modulation (trusted ref injected)
                # conf has num_streams ch; expand to feat_ch via repeat_interleave.
                conf_resized = F.interpolate(conf, size=x.size()[2:], mode=mode)
                if conf_resized.shape[1] != gamma.shape[1]:
                    repeat = gamma.shape[1] // conf_resized.shape[1]
                    conf_resized = conf_resized.repeat_interleave(repeat, dim=1)
                x = x * (1.0 + conf_resized * torch.tanh(gamma)) + conf_resized * beta
        
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
