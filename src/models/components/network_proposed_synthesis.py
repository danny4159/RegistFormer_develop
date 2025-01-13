import torch
import torch.nn as nn
import torch.nn.functional as F

class ProposedSynthesisModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        try:
            self.input_nc = kwargs['input_nc']
            self.feat_ch = kwargs['feat_ch']
            self.output_nc = kwargs['output_nc']
            self.demodulate = kwargs['demodulate']
            self.use_multiple_outputs = kwargs['use_multiple_outputs']

        except KeyError as e:
            raise ValueError(f"Missing required parameter: {str(e)}")
        
        if self.use_multiple_outputs: # For channel multiply
            ch = 2
        else:
            ch = 1
        
        # 일부분에만 style_denorm -> 21, 22, 31, 32에만 적용 #TODO: feat_ch에 ref ch만큼 배수로
        self.conv0 = StyleConv(self.input_nc, self.feat_ch * ch, kernel_size=3,
                                                 activate=True, demodulate=self.demodulate, ch=ch)
        self.conv11 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=True, activate=True, demodulate=self.demodulate, ch=ch)
        self.conv12 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=self.demodulate, ch=ch)
        self.conv21 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=True, activate=True, demodulate=self.demodulate, ch=ch)
        self.conv22 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=self.demodulate, ch=ch)
        self.conv31 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=self.demodulate, ch=ch)
        self.conv32 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=self.demodulate, ch=ch)
        self.conv41 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3, #feat_ch *4는 변치않게
                                upsample=True, activate=True, demodulate=self.demodulate, ch=ch)
        self.conv42 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                upsample=False, activate=True, demodulate=self.demodulate, ch=ch)
        self.conv51 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                upsample=True, activate=True, demodulate=self.demodulate, ch=ch)
        self.conv52 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                upsample=False, activate=True, demodulate=self.demodulate, ch=ch)
        self.conv6 = StyleConv(self.feat_ch * ch, self.feat_ch * ch, kernel_size=3,
                                activate=True, demodulate=self.demodulate, ch=ch) # 이걸 빠트렸었어.
                                # activate=False, demodulate=self.demodulate)
        
        # Added for checkerboard artifact
        if self.use_multiple_outputs:
            self.conv7_1 = nn.Conv2d(self.feat_ch * ch // 2, self.feat_ch * ch // 4, kernel_size=3, padding=1)
            self.conv7_2 = nn.Conv2d(self.feat_ch * ch // 2, self.feat_ch * ch // 4, kernel_size=3, padding=1)
        else:
            self.conv7 = nn.Conv2d(self.feat_ch * ch, self.feat_ch * ch // 2, kernel_size=3, padding=1)

        if self.use_multiple_outputs:
            self.conv8_1 = nn.Conv2d(self.feat_ch * ch // 4, self.feat_ch * ch // 8, kernel_size=3, padding=1)
            self.conv8_2 = nn.Conv2d(self.feat_ch * ch // 4, self.feat_ch * ch // 8, kernel_size=3, padding=1)
        else:
            self.conv8 = nn.Conv2d(self.feat_ch * ch // 2, self.feat_ch * ch // 4, kernel_size=3, padding=1)
                                       
        if self.use_multiple_outputs:
            self.conv_final_1 = nn.Conv2d(self.feat_ch * ch // 8, self.output_nc // 2, kernel_size=3, padding=1)
            self.conv_final_2 = nn.Conv2d(self.feat_ch * ch // 8, self.output_nc // 2, kernel_size=3, padding=1)
        else:
            self.conv_final = nn.Conv2d(self.feat_ch * ch // 4, self.output_nc, kernel_size=3, padding=1)

    def forward(self, merged_input, layers=[], encode_only=False):

        x = merged_input[:, :1, :, :]
        ref = merged_input[:, 1:, :, :]
        
        if self.training: # H,W = 128,128
            style_guidance_1 = F.interpolate(ref, scale_factor=1/16, mode='bilinear', align_corners=True)
        else: # H,W = 256,256
            style_guidance_1 = F.interpolate(ref, scale_factor=1/32, mode='bilinear', align_corners=True) # Final #TODO: sliding infer로 줄어든만큼 이것도 줄여줘야해. 8정도가 적절할듯 
        # style_guidance_1 = F.interpolate(ref, scale_factor=1/8, mode='bilinear', align_corners=True) # Ablation
        # style_guidance_1 = F.interpolate(ref, scale_factor=1/32, mode='bilinear', align_corners=True) # Ablation
        # style_guidance_1 = F.interpolate(ref, scale_factor=1/64, mode='bilinear', align_corners=True) # Ablation
        # style_guidance_1 = F.interpolate(ref, size=(1, 1), mode='bilinear', align_corners=True) # Ablation
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

        if self.use_multiple_outputs:
            feat6_1, feat6_2 = torch.chunk(feat6, chunks=2, dim=1)
            feat7_1 = self.conv7_1(feat6_1)
            feat8_1 = self.conv8_1(feat7_1)
            out_1 = self.conv_final_1(feat8_1)
            out_1 = torch.tanh(out_1)

            feat7_2 = self.conv7_2(feat6_2)
            feat8_2 = self.conv8_2(feat7_2)
            out_2 = self.conv_final_2(feat8_2)
            out_2 = torch.tanh(out_2)

            out = torch.cat((out_1, out_2), dim=1)
        else:
            feat7 = self.conv7(feat6)
            feat8 = self.conv8(feat7)
            out = self.conv_final(feat8)
            out = torch.tanh(out)

        if encode_only:
            # Collect intermediate features based on specified layers
            layers_dict = {
                0: feat0,
                1: feat1,
                2: feat2,
                3: feat3,
                4: feat4,
                5: feat5,
                6: feat6,
            }
            for i in layers:
                feats.append(layers_dict[i])
            return feats
        
        return out


class StyleConv(nn.Module):
    def __init__(self,
                 input_nc,
                 feat_ch,
                 kernel_size,
                 upsample=False,
                 downsample=False,
                 activate=False,
                 demodulate=True,
                 style_denorm=True,
                 eps=1e-8,
                 ch=1):
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

        if self.upsample:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(input_nc, feat_ch, kernel_size=3, padding=1)
            )
        
        elif self.downsample:
            self.down = nn.Sequential(
                nn.Conv2d(input_nc, feat_ch, stride=2, kernel_size=1, padding=0)
            )
        else:
            self.conv = nn.Conv2d(input_nc, feat_ch, kernel_size=3, padding=1)
            
        # self.batch_norm = nn.BatchNorm2d(feat_ch, affine=False)
        self.normalize = nn.InstanceNorm2d(feat_ch, affine=False)
        # self.batch_norm = nn.SyncBatchNorm(feat_ch, affine=False)

        nhidden = 512
        if ch == 1:
            self.mlp_shared = nn.Sequential( 
                nn.Conv2d(1, nhidden, kernel_size=3, padding=1),
                nn.ReLU())
            self.mlp_gamma = nn.Conv2d(nhidden, feat_ch, kernel_size=3, padding=1)
            self.mlp_beta = nn.Conv2d(nhidden, feat_ch, kernel_size=3, padding=1)
        elif ch == 2:
            self.mlp_shared = nn.Sequential( 
                nn.Conv2d(1, nhidden, kernel_size=3, padding=1),
                nn.ReLU())
            self.mlp_gamma = nn.Conv2d(nhidden, feat_ch//2, kernel_size=3, padding=1)
            self.mlp_beta = nn.Conv2d(nhidden, feat_ch//2, kernel_size=3, padding=1)

            self.mlp_shared_2 = nn.Sequential(
                nn.Conv2d(1, nhidden, kernel_size=3, padding=1),
                nn.ReLU())
            self.mlp_gamma_2 = nn.Conv2d(nhidden, feat_ch//2, kernel_size=3, padding=1)
            self.mlp_beta_2 = nn.Conv2d(nhidden, feat_ch//2, kernel_size=3, padding=1)

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
        
        x = self.normalize(x)
        
        # # Add noise
        if self.randomize_noise:
            noise = torch.randn_like(x) * self.noise_strength
        else:
            noise = torch.zeros_like(x) * self.noise_strength
        
        x = x + noise
        if self.style_denorm:
            # 1. style을 x의 크기와 맞게 interpolation 
            style = F.interpolate(style, size=x.size()[2:], mode='nearest') # TODO: bilinear 도 시도
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
    