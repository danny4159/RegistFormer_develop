import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import re
import numpy as np

###########################################################################################
# Generator
###########################################################################################

class SPADEGenerator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        try:
            self.num_upsample_layers = kwargs['num_upsample_layers']
            self.crop_size = kwargs['crop_size']
            self.aspect_ratio = kwargs['aspect_ratio']
            self.ngf = kwargs['ngf']
            self.use_vae = kwargs['use_vae']
            self.z_dim = kwargs['z_dim']
            self.norm_G = kwargs['norm_G']
            self.semantic_nc = kwargs['semantic_nc']
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {str(e)}")

        self.sw, self.sh = self.compute_latent_vector_size(self.num_upsample_layers, self.crop_size, self.aspect_ratio)

        if self.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(self.z_dim, 16 * self.ngf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.semantic_nc, 16 * self.ngf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * self.ngf, 16 * self.ngf, self.norm_G, self.semantic_nc)

        self.G_middle_0 = SPADEResnetBlock(16 * self.ngf, 16 * self.ngf, self.norm_G, self.semantic_nc)
        self.G_middle_1 = SPADEResnetBlock(16 * self.ngf, 16 * self.ngf, self.norm_G, self.semantic_nc)

        #TODO: 이부분 바꿈
        # self.up_0 = nn.Sequential(
        #     nn.Conv2d(16 * self.ngf, 8 * self.ngf, kernel_size=3, padding=1),
        #     nn.LeakyReLU(0.2, True)
        # )
        # self.up_1 = nn.Sequential(
        #     nn.Conv2d(8 * self.ngf, 4 * self.ngf, kernel_size=3, padding=1),
        #     nn.LeakyReLU(0.2, True)
        # )
        # self.up_2 = nn.Sequential(
        #     nn.Conv2d(4 * self.ngf, 2 * self.ngf, kernel_size=3, padding=1),
        #     nn.LeakyReLU(0.2, True)
        # )
        # self.up_3 = nn.Sequential(
        #     nn.Conv2d(2 * self.ngf, self.ngf, kernel_size=3, padding=1),
        #     nn.LeakyReLU(0.2, True)
        # )

        self.up_0 = SPADEResnetBlock(16 * self.ngf, 8 * self.ngf, self.norm_G, self.semantic_nc)
        self.up_1 = SPADEResnetBlock(8 * self.ngf, 4 * self.ngf, self.norm_G, self.semantic_nc)
        self.up_2 = SPADEResnetBlock(4 * self.ngf, 2 * self.ngf, self.norm_G, self.semantic_nc)
        self.up_3 = SPADEResnetBlock(2 * self.ngf, 1 * self.ngf, self.norm_G, self.semantic_nc)

        final_nc = self.ngf

        if self.num_upsample_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * self.ngf, self.ngf // 2, self.norm_G, self.semantic_nc)
            final_nc = self.ngf // 2

        # self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)
        self.conv_img = nn.Conv2d(final_nc, 1, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, num_upsample_layer, crop_size, aspect_ratio):#TODO: 사이즈 맞추기위해
    # def compute_latent_vector_size(self, num_upsample_layer, height, width):
        if num_upsample_layer == 'normal':
            num_up_layers = 5
        elif num_upsample_layer == 'more':
            num_up_layers = 6
        elif num_upsample_layer == 'most':
            num_up_layers = 7
        else:
            raise ValueError('num_upsample_layer [%s] not recognized' %
                             num_upsample_layer)

        #TODO: 사이즈 맞추기위해
        # sw = width // (2**num_up_layers)
        # sh = height // (2**num_up_layers)

        sw = crop_size // (2**num_up_layers)
        sh = round(sw / aspect_ratio)

        return sw, sh
    
    def forward(self, input, z=None):# [1, 1, 216, 256], [1, 256]
        #TODO: 사이즈 맞추기위해 추가한 코드
        # height, width = input.size(2), input.size(3)
        # self.sw, self.sh = self.compute_latent_vector_size(self.num_upsample_layers, height, width)

        if self.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z) # [1, 262144] = [1, 16 * ngf * sh * sw] 
            x = x.view(-1, 16 * self.ngf, self.sh, self.sw) # [1, 1024, 16, 16] = [1, 16 * ngf, sh, sw] 
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(input, size=(self.sh, self.sw))
            x = self.fc(x)

        x = self.head_0(x, input) # [1, 1024, 16, 16]

        x = self.up(x) # [1, 1024, 32, 32]
        x = self.G_middle_0(x, input) # # [1, 1024, 32, 32]

        if self.num_upsample_layers == 'more' or \
           self.num_upsample_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, input)

        #TODO: 이부분도 바꿈. up말고 나머지에 , input도 있었음 
        # x = self.up(x) # [1, 1024, 64, 64]
        # x = self.up_0(x) # [1, 512, 64, 64]
        # x = self.up(x) # [1, 512, 128, 128]
        # x = self.up_1(x)  # [1, 256, 128, 128]
        # x = self.up(x) # [1, 256, 256, 256]
        # x = self.up_2(x) # [1, 128, 256, 256]
        # x = self.up(x) # [1, 128, 512, 512]
        # x = self.up_3(x) # [1, 64, 512, 512]
        
        x = self.up(x) # [1, 1024, 64, 64]
        x = self.up_0(x, input) # [1, 512, 64, 64]
        x = self.up(x) # [1, 512, 128, 128]
        x = self.up_1(x, input)  # [1, 256, 128, 128]
        x = self.up(x) # [1, 256, 256, 256]
        x = self.up_2(x, input) # [1, 128, 256, 256]
        x = self.up(x) # [1, 128, 512, 512]
        x = self.up_3(x, input) # [1, 64, 512, 512]

        if self.num_upsample_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, input)

        #TODO: 내가 추가한 코드. 크기 맞추주기 위해
        x = F.interpolate(x, size=(input.size(2), input.size(3)), mode='bilinear', align_corners=False)

        x = self.conv_img(F.leaky_relu(x, 2e-1)) # [1, 1, 512, 512]
        x = F.tanh(x)

        return x


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, norm_G, semantic_nc):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, semantic_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = nn.SyncBatchNorm(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x) # 아무것도 변한게 없다..?

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out



###########################################################################################
# Enconder
###########################################################################################

class ConvEncoder(nn.Module):
    """ Same architecture as the image discriminator """

    def __init__(self, **kwargs):
        super().__init__()
        try:
            self.ngf = kwargs['ngf']
            self.norm_E = kwargs['norm_E']
            self.crop_size = kwargs['crop_size']
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {str(e)}")

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))

        norm_layer = get_nonspade_norm_layer(self.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(1, self.ngf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(self.ngf * 1, self.ngf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(self.ngf * 2, self.ngf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(self.ngf * 4, self.ngf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(self.ngf * 8, self.ngf * 8, kw, stride=2, padding=pw))
        if self.crop_size >= 256:
            self.layer6 = norm_layer(nn.Conv2d(self.ngf * 8, self.ngf * 8, kw, stride=2, padding=pw))

        self.so = s0 = 4
        self.fc_mu = nn.Linear(self.ngf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(self.ngf * 8 * s0 * s0, 256)

        self.actvn = nn.LeakyReLU(0.2, False)

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256: # TODO: 사이즈 맞춰주기위해
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        if self.crop_size >= 256:
            x = self.layer6(self.actvn(x))
        x = self.actvn(x)

        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x) # [1, 256]
        logvar = self.fc_var(x) # [1, 256]

        return mu, logvar


def get_nonspade_norm_layer(norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = nn.SyncBatchNorm(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer
