import torch
import torch.nn as nn
# from basicsr.utils.registry import ARCH_REGISTRY

import functools


class SynthesisMetaModule(nn.Module):
    def __init__(
        self,
        in_ch=1,
        out_ch=1,
        feat_ch=64,
        norm_layer=functools.partial(nn.InstanceNorm2d, affine=False),
        use_dropout=False,
        n_blocks=6,
        padding_type="reflect",
        requires_grad=False,
        load_path=None,
    ):
        assert n_blocks >= 0
        super(SynthesisMetaModule, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.feat_ch = feat_ch

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, feat_ch, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(feat_ch),
            nn.ReLU(True),
        ]

        n_downsampling = 2

        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(
                    feat_ch * mult,
                    feat_ch * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(feat_ch * mult * 2),
                nn.ReLU(True),
            ]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [
                ResnetBlock(
                    feat_ch * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            ]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            # model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
            #                             kernel_size=3, stride=2,
            #                             padding=1, output_padding=1,
            #                             bias=use_bias),
            model += [
                torch.nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(
                    feat_ch * mult,
                    int(feat_ch * mult / 2),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(int(feat_ch * mult / 2)),
                nn.ReLU(True),
            ]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(feat_ch, out_ch, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

        if load_path:
            checkpoint = torch.load(
                load_path, map_location=lambda storage, loc: storage
            )
            if "state_dict" in checkpoint:
                # PyTorch Lightning 형식의 .ckpt 파일
                model_state_dict = checkpoint["state_dict"]
                # 모델의 state_dict에 맞게 키 이름을 조정할 수 있습니다.
                adjusted_state_dict = {
                    k.replace("netG_A.", ""): v for k, v in model_state_dict.items()
                }
                self.load_state_dict(adjusted_state_dict, strict=False)
            else:
                # 일반적인 PyTorch 형식의 .pth 파일
                self.load_state_dict(checkpoint)

        if not requires_grad:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.train()
            for param in self.parameters():
                param.requires_grad = True

    def forward(self, input):
        return self.model(input)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias
        )

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


if __name__ == "__main__":
    height, width = 256, 256
    model = SynthesisMetaModule(in_ch=3, feat_ch=64, out_ch=3)
    print(model)

    src = torch.randn((2, 3, height, width))
    ref = torch.randn((2, 3, height, width))
    model.eval()
    with torch.no_grad():
        out = model(src, ref)
    model.train()

    print(out.shape)
