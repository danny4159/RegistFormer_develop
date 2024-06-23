import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops.upfirdn2d import upfirdn2d

class ProposedSynthesisModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        try:
            self.input_nc = kwargs['input_nc']
            self.feat_ch = kwargs['feat_ch']
            self.output_nc = kwargs['output_nc']
            self.demodulate = kwargs['demodulate']

        except KeyError as e:
            raise ValueError(f"Missing required parameter: {str(e)}")
        
        self.guide_net = nn.Sequential( # 이 conv들은 image의 style정보를 잘 대표하는 통계값이 나오도록 학습
            # 이것도 없었어 ver32는
            nn.Conv2d(self.input_nc, int(self.feat_ch/8), kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.Conv2d(self.feat_ch, self.feat_ch, kernel_size=3, stride=2, padding=1),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(int(self.feat_ch/8), int(self.feat_ch/8), kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.Conv2d(self.feat_ch, self.feat_ch, kernel_size=3, stride=2, padding=1),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(int(self.feat_ch/8), int(self.feat_ch/8), kernel_size=3, stride=1, padding=1), # 원래 여기까지
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.Conv2d(self.feat_ch, self.feat_ch, kernel_size=3, stride=2, padding=1),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.Conv2d(self.feat_ch, self.feat_ch, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.Conv2d(self.feat_ch, self.feat_ch, kernel_size=3, stride=2, padding=1),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.Conv2d(self.feat_ch, self.feat_ch, kernel_size=3, stride=1, padding=1),
            
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.AdaptiveAvgPool2d(1), # style transfer에서 많이 쓰이는 개념.
        )
        
        # 일부분에만 style_denorm -> 21, 22, 31, 32에만 적용
        self.conv0 = StyleConv(self.input_nc, self.feat_ch, kernel_size=3,
                                                 activate=True, demodulate=self.demodulate)
        self.conv11 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                downsample=True, activate=True, demodulate=self.demodulate)
        self.conv12 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=self.demodulate)
        self.conv21 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                downsample=True, activate=True, demodulate=self.demodulate)
        self.conv22 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=self.demodulate)
        self.conv31 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=self.demodulate)
        self.conv32 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=self.demodulate)
        self.conv41 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3, #feat_ch *4는 변치않게
                                upsample=True, activate=True, demodulate=self.demodulate)
        self.conv42 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                upsample=False, activate=True, demodulate=self.demodulate)
        self.conv51 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                upsample=True, activate=True, demodulate=self.demodulate)
        self.conv52 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                upsample=False, activate=True, demodulate=self.demodulate)
        self.conv6 = StyleConv(self.feat_ch, self.feat_ch, kernel_size=3,
                                activate=True, demodulate=self.demodulate) # 이걸 빠트렸었어.
                                # activate=False, demodulate=self.demodulate)


        self.conv_final = nn.Conv2d(self.feat_ch, self.output_nc, kernel_size=3, padding=1)

    def forward(self, x, ref, layers=[], encode_only=False):
        # style_guidance = self.guide_net(ref) # [1, 128, 24, 20]
        # ref = self.guide_net(ref) # [1, 128, 24, 20]
        # 원래는 interpolate 1/16만 했어. ver32
        style_guidance_1 = F.interpolate(ref, scale_factor=1/16, mode='bilinear', align_corners=True) # Original
        # style_guidance_2 = F.interpolate(ref, scale_factor=1/32, mode='bilinear', align_corners=True) # Original
        # style_guidance_3 = F.interpolate(ref, scale_factor=1/64, mode='bilinear', align_corners=True) # Original
        # style_guidance = F.interpolate(ref, scale_factor=1/32, mode='bilinear', align_corners=True)
        # style_guidance = F.interpolate(ref, scale_factor=1/4, mode='bilinear', align_corners=True)
        # style_guidance = F.adaptive_avg_pool2d(ref, (24, 20))

        feats = []
        feat0 = self.conv0(x, style_guidance_1) # [1, 64, 384, 320]
        feat1 = self.conv11(feat0, style_guidance_1) # [1, 64, 192, 160]
        feat1 = self.conv12(feat1, style_guidance_1) # [1, 128, 192, 160]
        feat2 = self.conv21(feat1, style_guidance_1) # [1, 128, 96, 80]
        feat2 = self.conv22(feat2, style_guidance_1) # [1, 256, 96, 80]
        feat3 = self.conv31(feat2, style_guidance_1) # [1, 256, 96, 80]
        feat3 = self.conv32(feat3, style_guidance_1) # [1, 256, 96, 80]
        feat4 = self.conv41(feat3 + feat2, style_guidance_1) # [1, 128, 192, 160]
        feat4 = self.conv42(feat4, style_guidance_1) # [1, 128, 192, 160]
        feat5 = self.conv51(feat4 + feat1, style_guidance_1) # [1, 64, 384, 320]
        feat5 = self.conv52(feat5, style_guidance_1) # [1, 64, 384, 320]
        feat6 = self.conv6(feat5 + feat0, style_guidance_1) # [1, 64, 384, 320]
        out = self.conv_final(feat6) # [1, 1, 384, 320]
        out = torch.tanh(out) # [1, 1, 384, 320]
        
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
                 blur_kernel=[1, 1.5, 1.5, 1],
                #  blur_kernel=[1, 2, 2, 1],
                #  blur_kernel=[1, 3, 3, 1],
                 demodulate=True,
                 style_denorm=True,
                 eps=1e-8,):
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
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, (pad0, pad1), upsample_factor=factor)
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(input_nc, feat_ch, kernel_size=3, padding=1)
            )
        
        elif self.downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(blur_kernel, (pad0, pad1))
            self.down = nn.Sequential(
                nn.Conv2d(input_nc, feat_ch, stride=2, kernel_size=1, padding=0)
            )
        else:
            self.conv = nn.Conv2d(input_nc, feat_ch, kernel_size=3, padding=1)
            
        # self.batch_norm = nn.BatchNorm2d(feat_ch, affine=False)
        self.normalize = nn.InstanceNorm2d(feat_ch, affine=False)
        # self.batch_norm = nn.SyncBatchNorm(feat_ch, affine=False)

        nhidden = 512
        self.mlp_shared = nn.Sequential(
            # nn.Conv2d(feat_ch, nhidden, kernel_size=3, padding=1),
            # nn.Conv2d(int(self.feat_ch/8), nhidden, kernel_size=3, padding=1),
            nn.Conv2d(1, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, feat_ch, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, feat_ch, kernel_size=3, padding=1)

        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.randomize_noise = True
        self.noise_strength = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x, style): # style: [1, 64, 1, 1]
        
        if self.downsample:
            original_size = x.size()
            # x = self.blur(x)
            if x.size()[2:] != original_size[2:]:  # If the size has changed
                x = F.interpolate(x, size=original_size[2:], mode='bilinear', align_corners=True)
            x = self.down(x)
        elif self.upsample:
            x = self.up(x)
            original_size = x.size()
            # x = self.blur(x)
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
            # 1. style을 x의 크기와 맞게 interpolation #TODO: Interpolation을 할지 브로드캐스트를 할지
            style = F.interpolate(style, size=x.size()[2:], mode='nearest') # TODO: bilinear 도 시도
            
            # 2. style을 x의 C과 맞게 mlp
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