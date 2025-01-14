import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
from torch.autograd import Variable
import math
import pdb
import os
from flash_attn import flash_attn_func
from src.models.components.network_voxelmorph_original import VecInt
from src.models.components.performer import CrossAttention


class RegistFormer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        try:
            self.feat_dim = kwargs['feat_dim']
            self.ref_ch = kwargs['ref_ch']
            self.src_ch = kwargs['src_ch']
            self.out_ch = kwargs['out_ch']
            self.nhead = kwargs['nhead']
            self.mlp_ratio = kwargs['mlp_ratio']
            self.pos_en_flag = kwargs['pos_en_flag']
            self.k_size = kwargs['k_size']
            self.attn_type = kwargs['attn_type']
            self.daca_loc = kwargs['daca_loc']
            self.flow_type = kwargs['flow_type']
            self.dam_type = kwargs['dam_type']
            self.fuse_type = kwargs['fuse_type']
            self.flow_model_path = kwargs['flow_model_path']
            self.flow_ft = kwargs['flow_ft']
            self.flow_size = kwargs.get('flow_size', None)
            self.dam_ft = kwargs['dam_ft']
            self.dam_path = kwargs['dam_path']
            self.dam_feat = kwargs['dam_feat']
            self.main_ft = kwargs['main_ft']
            self.is_moved_feat = kwargs['is_moved_feat']

        except KeyError as e:
            raise ValueError(f"Missing required parameter: {str(e)}")
        
        current_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

        self.flow_model_path = os.path.join(project_root, self.flow_model_path)
        self.dam_path = os.path.join(project_root, self.dam_path)

        if self.flow_type == "voxelmorph":
            from src.models.components.voxelmorph import VxmDense
            self.flow_estimator = VxmDense.load(path=self.flow_model_path, device='cpu')
            self.flow_estimator.eval()

        elif self.flow_type == "voxelmorph_lightning" or self.flow_type == "zero":
            from src.models.components.network_voxelmorph_original import VxmDense
            self.flow_estimator = VxmDense(inshape=self.flow_size,
                                           nb_unet_features=[[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]],
                                           nb_unet_levels=None,
                                           unet_feat_mult=1,
                                           nb_unet_conv_per_level=1,
                                           int_steps=7,
                                           int_downsize=2,
                                           bidir=False,
                                           use_probs=False,
                                           src_feats=self.src_ch,
                                           trg_feats=self.ref_ch,
                                           unet_half_res=False,)
            checkpoint = torch.load(self.flow_model_path, map_location=lambda storage, loc: storage)
            model_state_dict = checkpoint["state_dict"]
            adjusted_state_dict = {k.replace("netR_A.", ""): v for k, v in model_state_dict.items()}
            self.flow_estimator.load_state_dict(adjusted_state_dict, strict=False)
            self.flow_estimator.eval()

        elif self.flow_type == "grad_icon":
            from src.models.components.network_grad_icon import GradICON
            self.flow_estimator = GradICON(batch_size=1, dimension=2, input_size=[384,320])
            checkpoint = torch.load(self.flow_model_path, map_location=lambda storage, loc: storage)
            model_state_dict = checkpoint["state_dict"]
            adjusted_state_dict = {k.replace("netR_A.", ""): v for k, v in model_state_dict.items()}
            self.flow_estimator.load_state_dict(adjusted_state_dict, strict=False)
            self.flow_estimator.eval()
            
        else:
            raise ValueError(f"Unrecognized flow type: {self.flow_type}.")

        for param in self.flow_estimator.parameters():
            param.requires_grad = self.flow_ft  # False
            
        # Define DAM.
        if self.dam_ft:
            assert self.dam_path != None
        
        if self.dam_type == "synthesis_meta":
            from src.models.components.meta_synthesis import (
                SynthesisMetaModule,
            )   

            self.DAM = SynthesisMetaModule(
                in_ch=self.src_ch,
                feat_ch=self.dam_feat,
                out_ch=self.src_ch,
                load_path=self.dam_path,
                requires_grad=self.dam_ft,
            )

        elif self.dam_type == "proposed_synthesis":
            from src.models.components.network_proposed_synthesis import ProposedSynthesisModule
            self.DAM = ProposedSynthesisModule(input_nc=1, feat_ch=256, output_nc=1, demodulate=True)
            checkpoint = torch.load(self.dam_path, map_location=lambda storage, loc: storage)
            model_state_dict = checkpoint["state_dict"]
            adjusted_state_dict = {k.replace("netG_A.", ""): v for k, v in model_state_dict.items()}
            self.DAM.load_state_dict(adjusted_state_dict, strict=False)
            self.DAM.eval()
            for param in self.DAM.parameters():
                param.requires_grad = self.dam_ft

        elif self.dam_type == "munit":
            from src.models.components.network_adainGen import AdaINGen
            self.DAM_A = AdaINGen(input_nc=1, output_nc=1, ngf=64)
            checkpoint = torch.load(self.dam_path, map_location=lambda storage, loc: storage)
            model_state_dict = checkpoint["state_dict"]
            adjusted_state_dict = {k.replace("netG_A.", ""): v for k, v in model_state_dict.items()}
            self.DAM_A.load_state_dict(adjusted_state_dict, strict=False)
            self.DAM_A.eval()
            for param in self.DAM_A.parameters():
                param.requires_grad = self.dam_ft

            self.DAM_B = AdaINGen(input_nc=1, output_nc=1, ngf=64)
            checkpoint = torch.load(self.dam_path, map_location=lambda storage, loc: storage)
            model_state_dict = checkpoint["state_dict"]
            adjusted_state_dict = {k.replace("netG_B.", ""): v for k, v in model_state_dict.items()}
            self.DAM_B.load_state_dict(adjusted_state_dict, strict=False)
            self.DAM_B.eval()
            for param in self.DAM_B.parameters():
                param.requires_grad = self.dam_ft

        # Define feature extractor.
        # self.unet_q = Unet(src_ch, feat_dim, feat_dim)
        self.unet_q = Unet(self.src_ch * 2, self.feat_dim, self.feat_dim)
        self.unet_k = Unet(self.ref_ch, self.feat_dim, self.feat_dim)

        # Define GAM.
        self.trans_unit = nn.ModuleList(
            [
                TransformerUnit(
                    self.feat_dim,
                    self.nhead,
                    self.pos_en_flag,
                    self.mlp_ratio,
                    self.k_size,
                    self.attn_type,
                    self.fuse_type,
                ),
                TransformerUnit(
                    self.feat_dim,
                    self.nhead,
                    self.pos_en_flag,
                    self.mlp_ratio,
                    self.k_size,
                    self.attn_type,
                    self.fuse_type,
                ),
                TransformerUnit(
                    self.feat_dim,
                    self.nhead,
                    self.pos_en_flag,
                    self.mlp_ratio,
                    self.k_size,
                    self.attn_type,
                    self.fuse_type,
                ),
            ]
        )

        self.conv0 = double_conv(self.feat_dim, self.feat_dim)
        self.conv1 = double_conv_down(self.feat_dim, self.feat_dim)
        self.conv2 = double_conv_down(self.feat_dim, self.feat_dim)
        self.conv3 = double_conv(self.feat_dim, self.feat_dim)
        self.conv4 = double_conv_up(self.feat_dim, self.feat_dim)
        self.conv5 = double_conv_up(self.feat_dim, self.feat_dim)
        self.conv6 = nn.Sequential(
            single_conv(self.feat_dim, self.feat_dim), nn.Conv2d(self.feat_dim, self.out_ch, 3, 1, 1)
        )

        if not self.main_ft:
            self.eval()
            for key, param in self.named_parameters():
                if "flow_estimator" not in key and "DAM" not in key:
                    param.requires_grad = False
        else:
            self.train()
            for key, param in self.named_parameters():
                if "flow_estimator" not in key and "DAM" not in key:
                    param.requires_grad = True

        # VecInt initial
        # self.int_steps = kwargs.get('int_steps', 7)  # 적분 단계 수 (기본값 7)
        # self.vecint = VecInt(self.flow_size, self.int_steps)


    # def forward(self, src, ref, mask=None, for_nce=False, for_src=False):
    def forward(self, merged_input, mask=None, for_nce=False, for_src=False):

        channels = merged_input.shape[1] // 2 # FIXME: 이거 ref 채널에 따라 맞게 2,3 다를수있어
        src = merged_input[:, :channels, :, :]
        ref = merged_input[:, channels:, :, :]
        
        assert (
            src.shape == ref.shape
        ), "Shapes of source and reference images mismatch. Source shape: {src.shape}, Reference shape: {ref.shape}"
        device = src.device
        self.flow_estimator.to(device)
        moved = None

        # if not self.training:
        #     N, C, H, W = src.shape
        #     mod_size = 4
        #     H_pad = mod_size - H % mod_size if not H % mod_size == 0 else 0
        #     W_pad = mod_size - W % mod_size if not W % mod_size == 0 else 0
        #     src = F.pad(src, (0, W_pad, 0, H_pad), "replicate")
        #     ref = F.pad(ref, (0, W_pad, 0, H_pad), "replicate")

        if self.dam_type == "dam" or self.dam_type == "proposed_synthesis":
            src_origin = src
            src = self.DAM(src, ref)  # [4, 3, 256, 256]
        elif self.dam_type == "synthesis_meta":
            src_origin = src
            src = self.DAM(src)
        elif self.dam_type == "dam_misalign":
            src_origin = src
            ref_to_src = self.DAM_MR_CT(ref, src)
            src = self.DAM(src, ref)
        # elif self.dam_type == 'dam_misalign':
        #     src = self.DAM.netG_B(src, ref)
        elif self.dam_type == "munit":
            src_origin = src
            c_a, s_a = self.DAM_A.encode(src)
            c_b, s_b = self.DAM_B.encode(ref)
            src = self.DAM_B.decode(c_a, s_b)
        else:
            raise ValueError(
                "Invalid dam_type provided. Expected 'dam' or 'synthesis_meta'."
            )

        if for_nce:
            if for_src:
                src_lq_concat = torch.cat((src_origin, src), dim=1)
                q_feat = self.unet_q(src_lq_concat, for_nce=True)
                return q_feat
            else:
                q_feat = self.unet_k(ref, for_nce=True)
                return q_feat


        height_multiple = self.flow_size[0] if self.flow_size else 768
        width_multiple = self.flow_size[1] if self.flow_size else 576

        if self.flow_type in ["voxelmorph", "zero","voxelmorph_lightning"]:
            if self.dam_type == "synthesis_meta" or self.dam_type == "dam" or self.dam_type == "proposed_synthesis" or self.dam_type == "munit":
                src, moving_padding = self.pad_tensor_to_multiple(src, height_multiple=height_multiple, width_multiple=width_multiple)
                ref, fixed_padding = self.pad_tensor_to_multiple(ref, height_multiple=height_multiple, width_multiple=width_multiple)
                
                if self.flow_type == "zero":
                    moved, flow = self.flow_estimator(ref, src, registration=True)  # Zeroflow version
                else:
                    _, flow = self.flow_estimator(src, ref, registration=True)  # Original version # 첫번째 입력변수가 moving, 두번째 입력변수가 fixed

                    # flow = self.vecint(flow) #TODO: modified

                src = self.crop_tensor_to_original(src, fixed_padding)
                ref = self.crop_tensor_to_original(ref, fixed_padding)
                flow = self.crop_tensor_to_original(flow, fixed_padding)
                if self.flow_type == "zero":
                    flow = torch.zeros_like(flow)  # Zeroflow version
                if self.is_moved_feat: # Comparison
                    moved = self.crop_tensor_to_original(moved, fixed_padding)

            elif self.dam_type == "dam_misalign":
                src_origin, moving_padding = self.pad_tensor_to_multiple(src_origin, height_multiple=height_multiple, width_multiple=width_multiple)
                ref_to_src, fixed_padding = self.pad_tensor_to_multiple(ref_to_src, height_multiple=height_multiple, width_multiple=width_multiple)
                ref, fixed_padding = self.pad_tensor_to_multiple(ref, height_multiple=height_multiple, width_multiple=width_multiple)

                _, flow = self.flow_estimator(src_origin, ref_to_src, registration=True)
                _, flow_mr = self.flow_estimator(ref_to_src, src_origin, registration=True)
                ref = self.crop_tensor_to_original(ref, fixed_padding)
                flow = self.crop_tensor_to_original(flow, fixed_padding)
            else:
                raise ValueError("Invalid dam_type")
        elif self.flow_type == 'grad_icon':
            height = src.shape[2]
            batch_size = src.shape[0]
            flows = []
            src = self.pad_height_to_384(src)
            ref = self.pad_height_to_384(ref)
            src, moving_padding = self.pad_tensor_to_multiple(src, height_multiple=384, width_multiple=320)
            ref, fixed_padding = self.pad_tensor_to_multiple(ref, height_multiple=384, width_multiple=320)
            for i in range(batch_size): #Because of flow_estimator is trained on batch 1
                src_i = src[i:i+1]
                ref_i = ref[i:i+1]
                _, flow_i, _ = self.flow_estimator(src_i, ref_i)
                flows.append(flow_i)
            flow = torch.cat(flows, dim=0)
            src = self.crop_height_to_original(src, height)
            ref = self.crop_height_to_original(ref, height)
            flow = self.crop_height_to_original(flow, height)
            src = self.crop_tensor_to_original(src, fixed_padding)
            ref = self.crop_tensor_to_original(ref, fixed_padding)
            flow = self.crop_tensor_to_original(flow, fixed_padding)
        else:
            flow = self.flow_estimator(src, ref)  # src가 이동 ref가 고정

        src_lq_concat = torch.cat((src_origin, src), dim=1)
        # q_feat = self.unet_q(src)
        q_feat = self.unet_q(src_lq_concat)  # 발전시킨것

        if self.is_moved_feat:
            k_feat = self.unet_k(moved) # 이건 zeroflow와 함께. moved를 key,value로 쓰기위해. 결과 꽤 좋더라?
        else:
            k_feat = self.unet_k(ref)

        outputs = []
        for i in range(3):
            # if i == 2:
            if (
                self.daca_loc == "end" and i != 2
            ):  # end일땐 끝에거만 cross-attention하도록
                continue

            if mask != None:
                mask = mask[:, 0:1, :, :]
                outputs.append(
                    self.trans_unit[i](
                        q_feat[i + 3], k_feat[i + 3], k_feat[i + 3], flow, mask
                    )
                )
            else:
                outputs.append(
                    self.trans_unit[i](
                        q_feat[i + 3], k_feat[i + 3], k_feat[i + 3], flow
                    )
                )

        if self.daca_loc == "end":
            f0 = self.conv0(outputs[0])  # H, W
            f1 = self.conv1(f0)  # H/2, W/2
            f2 = self.conv2(f1)  # H/4, W/4
            f3 = self.conv3(f2)  # H/4, W/4
            f3 = f3 + f2
            f4 = self.conv4(f3)  # H/2, W/2
            f4 = f4 + f1
            f5 = self.conv5(f4)  # H, W
            f5 = f5 + outputs[0] + f0

        else:
            f0 = self.conv0(outputs[2])  # H, W
            f1 = self.conv1(f0)  # H/2, W/2
            f1 = f1 + outputs[1]
            f2 = self.conv2(f1)  # H/4, W/4
            f2 = f2 + outputs[0]
            f3 = self.conv3(f2)  # H/4, W/4
            f3 = f3 + outputs[0] + f2
            f4 = self.conv4(f3)  # H/2, W/2
            f4 = f4 + outputs[1] + f1
            f5 = self.conv5(f4)  # H, W
            f5 = f5 + outputs[2] + f0

        out = self.conv6(f5)
        out = torch.tanh(out)  # 내가 추가한 코드

        # if not self.training:
        #     out = out[:, :, :H, :W]

        # if moved is None:
        #     return out, src, out
        # else:
        #     return out, src, moved #src는 dam 결과
        return out


    def pad_tensor_to_multiple(self, tensor, height_multiple, width_multiple):
        _, _, h, w = tensor.shape
        h_pad = (height_multiple - h % height_multiple) % height_multiple
        w_pad = (width_multiple - w % width_multiple) % width_multiple

        # Pad the tensor
        padded_tensor = F.pad(
            tensor, (0, w_pad, 0, h_pad), mode="constant", value=-1
        )

        return padded_tensor, (h_pad, w_pad)

    def crop_tensor_to_original(self, tensor, padding):
        h_pad, w_pad = padding
        return tensor[:, :, : tensor.shape[2] - h_pad, : tensor.shape[3] - w_pad]
        
    def pad_height_to_384(self, tensor, padding_value=-1):
        # tensor shape: [batch, channel, height, width, slice]
        height = tensor.shape[2]
        if height < 384:
            # padding = (0, 384 - height)  # padding only on one side
            padding = (0, 0, 0, 384 - height)
            tensor = F.pad(tensor, padding, mode='constant', value=padding_value)
        return tensor
    
    def crop_height_to_original(self, tensor, original_height):
        # tensor shape: [batch, channel, height, width, slice]
        return tensor[:, :, :original_height, :]








####################################################################################################
####################################################################################################











def resize_flow(flow, size_type, sizes, interp_mode="bilinear", align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == "ratio":
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == "shape":
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(
            f"Size type should be ratio or shape, but got type {size_type}."
        )

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow,
        size=(output_h, output_w),
        mode=interp_mode,
        align_corners=align_corners,
    )
    return resized_flow


def softmax_attention(q, k, v):
    # n x 1(k^2) x nhead x d x h x w
    h, w = q.shape[-2], q.shape[-1]

    q = q.permute(0, 2, 4, 5, 1, 3)  # n, nhead, h, w, 1, d # 지금 보면 n,head,h,w가 다 보존되고 있는거야. 그에 독립적으로 d차원 벡터가 있는거.
    k = k.permute(0, 2, 4, 5, 3, 1)  # n, nhead, h, w, d, k^2
    v = v.permute(0, 2, 4, 5, 1, 3)  # n, nhead, h, w, k^2, d

    N = q.shape[-1]  # scaled attention
    attn = torch.matmul(
        q / N**0.5, k
    )  # TODO: 이때 validation 튄다
    attn = F.softmax(attn, dim=-1)
    output = torch.matmul(attn, v) # n, nhead, h, w, 1, d

    output = output.permute(0, 4, 1, 5, 2, 3).squeeze(1) # n, nhead, d, h, w
    attn = attn.permute(0, 4, 1, 5, 2, 3).squeeze(1) 

    return output, attn


# temporal for global attention.
def dotproduct_attention(q, k, v):
    # b x n x d x h x w
    h, w = q.shape[-2], q.shape[-1]

    q = q.flatten(-2).transpose(-2, -1)  # b x n x hw x d
    k = k.flatten(-2)  # b x n x d x hw
    v = v.flatten(-2).transpose(-2, -1)  # b x n x hw x d

    N = k.shape[-1]
    attn = None
    tmp = torch.matmul(k, v) / N
    output = torch.matmul(q, tmp)

    output = output.transpose(-2, -1)
    output = output.view(*output.shape[:-1], h, w)

    return output, attn

def performer_attention(q, k, v, performer_cross_attn = None, nb_features=None, projection_matrix=None, eps=1e-6):
    # n x 1(k^2) x nhead x d x h x w
    n, _, nhead, d, h, w = q.shape
    _, kk, _, _, _, _ = k.shape

    q = q.permute(0, 2, 4, 5, 1, 3)  # n, nhead, h, w, 1, d
    k = k.permute(0, 2, 4, 5, 1, 3)  # n, nhead, h, w, k^2, d
    v = v.permute(0, 2, 4, 5, 1, 3)  # n, nhead, h, w, k^2, d
    
    # q = q.reshape(n, h * w * 1, nhead * d) # n, h * w * 1, nhead * d
    # k = k.reshape(n,  h * w * kk, nhead * d) # n, h * w * k^2, nhead * d
    # v = v.reshape(n,  h * w * kk, nhead * d) # n, h * w * k^2, nhead * d

    output = performer_cross_attn(q, k, v, context=k) # n, h * w * 1, nhead * d  /New: n, nhead, h, w, 1, d

    output = output.permute(0, 1, 5, 2, 3, 4).squeeze(-1) # n, nhead, d, h, w

    return output

def flash_attention(q, k, v):
    # k = k.to(torch.bfloat16)
    # v = v.to(torch.bfloat16)
    # print("#############################################################")
    # print(f"q dtype: {q.dtype}, k dtype: {k.dtype}, v dtype: {v.dtype}")
    # print("#############################################################")
    # height, width = q.shape[4], q.shape[5]
    # q = q.permute(0, 2, 4, 5, 1, 3)  # (batch_size, head, height, width, 1, dim)
    # q = q.reshape(q.shape[0], q.shape[1], q.shape[2] * q.shape[3], q.shape[5])  # (batch_size, head, height * width, dim)

    # k = k.permute(0, 2, 4, 5, 3, 1)  # (batch_size, head, height, width, dim, k^2)
    # k = k.reshape(k.shape[0], k.shape[1], k.shape[2] * k.shape[3], k.shape[4], k.shape[5])  # (batch_size, head, height * width, dim, k_sq)
    # k = k.permute(0, 1, 2, 4, 3)  # (batch_size, head, height * width, k^2, dim)

    # v = v.permute(0, 2, 4, 5, 1, 3)  # (batch_size, head, height, width, k^2, dim)
    # v = v.reshape(v.shape[0], v.shape[1], v.shape[2] * v.shape[3], v.shape[4], v.shape[5])  # (batch_size, head, height * width, k_sq, dim)

    # output = flash_attn_func(q=q,k=k,v=v)

    # output = output.reshape(output.shape[0], output.shape[1], height, width, output.shape[3])
    # output = output.permute(0, 1, 4, 2, 3)  # (batch_size, head, dim, height, width)
    # output = output.reshape(output.shape[0], 1, output.shape[1], output.shape[2], output.shape[3], output.shape[4])

    k = k.to(torch.bfloat16)
    v = v.to(torch.bfloat16)
    q = q.to(torch.bfloat16)

    # Check the expected dimensions
    batch_size = q.shape[0]
    num_heads = q.shape[1]
    seqlen_q = q.shape[2] * q.shape[3]  # Assume q's last two dims are h and w
    seqlen_kv = k.shape[2] * k.shape[3]  # Similarly for k and v

    head_size = q.shape[-1]

    # Reshape q, k, v
    q = q.permute(0, 2, 3, 1, 4).reshape(batch_size, seqlen_q, num_heads, head_size)  # (batch_size, seqlen_q, num_heads, head_size)
    k = k.permute(0, 2, 3, 1, 4).reshape(batch_size, seqlen_kv, num_heads, head_size)  # (batch_size, seqlen_kv, num_heads, head_size)
    v = v.permute(0, 2, 3, 1, 4).reshape(batch_size, seqlen_kv, num_heads, head_size)  # (batch_size, seqlen_kv, num_heads, head_size)

    # Print shapes for debugging
    print(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")

    # Apply flash attention
    output = flash_attn_func(q=q, k=k, v=v)

    # Reshape output back to original format
    output = output.reshape(batch_size, seqlen_q, num_heads, head_size)
    output = output.permute(0, 2, 1, 3).reshape(batch_size, num_heads, head_size, q.shape[2], q.shape[3])

    return output

class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class double_conv_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class double_conv_up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv_up, self).__init__()
        self.conv = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class PosEnSine(nn.Module):
    """
    Code borrowed from DETR: models/positional_encoding.py
    output size: b*(2.num_pos_feats)*h*w
    """

    def __init__(self, num_pos_feats):
        super(PosEnSine, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.normalize = True
        self.scale = 2 * math.pi
        self.temperature = 10000

    def forward(self, x):
        b, c, h, w = x.shape
        not_mask = torch.ones(1, h, w, device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.repeat(b, 1, 1, 1)
        return pos


class MLP(nn.Module):
    """
    conv-based MLP layers.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.linear1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.linear2 = nn.Conv2d(hidden_features, out_features, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


class TransformerUnit(nn.Module):
    def __init__(
        self,
        feat_dim,
        n_head=8,
        pos_en_flag=True,
        mlp_ratio=2,
        k_size=5,
        attn_type="softmax",
        fuse_type=None,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.attn_type = attn_type
        self.fuse_type = fuse_type
        self.pos_en_flag = pos_en_flag

        self.pos_en = PosEnSine(self.feat_dim // 2)
        self.attn = MultiheadAttention(feat_dim, n_head, k_size=k_size, attn_type=self.attn_type)

        mlp_hidden_dim = int(feat_dim * mlp_ratio)
        self.mlp = MLP(in_features=feat_dim, hidden_features=mlp_hidden_dim)
        self.norm = nn.GroupNorm(1, self.feat_dim)

        if fuse_type:
            if fuse_type == "conv":
                self.fuse_conv = double_conv(in_ch=feat_dim, out_ch=feat_dim)
            elif fuse_type == "mask":
                self.fuse_conv = double_conv(in_ch=feat_dim, out_ch=feat_dim)

    def forward(self, q, k, v, flow, mask=None):
        if q.shape[-2:] != flow.shape[-2:]:
            # pdb.set_trace()
            flow = resize_flow(flow, "shape", q.shape[-2:])
        if mask != None and q.shape[-2:] != mask.shape[-2:]:
            # pdb.set_trace()
            mask = F.interpolate(mask, size=q.shape[-2:], mode="nearest")
        if self.pos_en_flag:
            q_pos_embed = self.pos_en(q)
            k_pos_embed = self.pos_en(k)
        else:
            q_pos_embed = 0
            k_pos_embed = 0

        # print(flow)
        # cross-multi-head attention
        # out, attn = checkpoint(self._attn_forward, q + q_pos_embed, k + k_pos_embed, v, flow)
        out, attn = self.attn(
            q=q + q_pos_embed,
            k=k + k_pos_embed,
            v=v,
            flow=flow,
            attn_type=self.attn_type,
        )
        # print(attn.shape)

        if self.fuse_type:
            if self.fuse_type == "conv":
                out = out + self.fuse_conv(q)
            elif self.fuse_type == "mask":
                try:
                    assert mask != None, "No mask found."
                except:
                    pdb.set_trace()
                out = (1 - mask) * out + mask * self.fuse_conv(q)

        # feed forward
        out = out + self.mlp(out)
        out = self.norm(out)

        return out


class Unet(nn.Module):
    def __init__(self, in_ch, feat_ch, out_ch):
        super().__init__()
        self.conv_in = single_conv(in_ch, feat_ch)

        self.conv1 = double_conv_down(feat_ch, feat_ch)
        self.conv2 = double_conv_down(feat_ch, feat_ch)
        self.conv3 = double_conv(feat_ch, feat_ch)
        self.conv4 = double_conv_up(feat_ch, feat_ch)
        self.conv5 = double_conv_up(feat_ch, feat_ch)
        self.conv6 = double_conv(feat_ch, out_ch)

    def forward(self, x, for_nce=False):
        feat0 = self.conv_in(x)  # H, W
        feat1 = self.conv1(feat0)  # H/2, W/2
        feat2 = self.conv2(feat1)  # H/4, W/4
        feat3 = self.conv3(feat2)  # H/4, W/4
        feat3 = feat3 + feat2  # H/4
        feat4 = self.conv4(feat3)  # H/2, W/2
        feat4 = feat4 + feat1  # H/2, W/2
        feat5 = self.conv5(feat4)  # H
        feat5 = feat5 + feat0  # H
        feat6 = self.conv6(feat5)
        if for_nce:
            return [feat0, feat1, feat2, feat3, feat4, feat5, feat6]
        return feat0, feat1, feat2, feat3, feat4, feat6


class MultiheadAttention(nn.Module):
    def __init__(self, feat_dim, n_head, k_size=5, d_k=None, d_v=None, attn=None, attn_type="softmax"):
        super().__init__()
        if d_k is None:
            d_k = feat_dim // n_head
        if d_v is None:
            d_v = feat_dim // n_head

        self.n_head = n_head
        self.k_size = k_size
        self.d_k = d_k
        self.d_v = d_v

        # pre-attention projection
        self.w_qs = nn.Conv2d(feat_dim, n_head * d_k, 1, bias=False)
        self.w_ks = nn.Conv2d(feat_dim, n_head * d_k, 1, bias=False)
        self.w_vs = nn.Conv2d(feat_dim, n_head * d_v, 1, bias=False)

        # after-attention combine heads
        self.fc = nn.Conv2d(n_head * d_v, feat_dim, 1, bias=False)

        # Performer
        if attn_type == "performer":
            self.performer_cross_attn = CrossAttention(dim = 8, heads = 2) #(dim = 14, heads = 2)

        # self.linear_k_reduce = nn.Linear(self.k_size**2, self.k_size**2 // 8, bias=False)
        # self.linear_v_reduce = nn.Linear(self.k_size**2, self.k_size**2 // 8, bias=False)

        # self.k_reduce_conv = nn.Conv2d(
        #     in_channels=k_size**2,  # Input channels: grid size
        #     out_channels=k_size**2 // 4,  # Output channels (adjust as needed)
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        #     bias=True,  # Ensure learnability
        # )
        # self.v_reduce_conv = nn.Conv2d(
        #     in_channels=k_size**2,
        #     out_channels=k_size**2 // 4,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        #     bias=True,
        # )

    def forward(self, q, k, v, flow, attn_type="softmax"):
        # input: n x c x h x w
        # flow: n x 2 x h x w
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        # Pass through the pre-attention projection:
        # n x c x h x w   ---->   n x (nhead*dk) x h x w  (c = nhead*dk)
        q = self.w_qs(q)  # [2, 14, 24, 24]
        k = self.w_ks(k)
        v = self.w_vs(v)  # [2, 14, 24, 24]

        n, c, h, w = q.shape

        # ------ Sampling K and V features ---------
        sampling_grid = flow_to_grid(flow, self.k_size)  # return: src에서 각 local grid가 어떻게 움직여야 ref local grid가 되는지 # n x k**2, h, w, 2  # # [100, 48, 48, 2]
        # sampled feature
        # n x k^2 x c x h x w
        sample_k_feat = flow_guide_sampler(k, sampling_grid, k_size=self.k_size)  # [4, 25, 64, 48, 48] # ref에서 feature map의 local grid
        sample_v_feat = flow_guide_sampler(v, sampling_grid, k_size=self.k_size)

        # Reshape for multi-head attention.
        # q: n x 1 x nhead x dk x h x w
        # k,v: n x k^2 x nhead x dk x h x w
        q = q.view(n, 1, n_head, d_k, h, w)  # [2, 1, 2, 7, 24, 24]
        k = sample_k_feat.view(n, self.k_size**2, n_head, d_k, h, w)  # [2, 784, 2, 7, 24, 24]
        v = sample_v_feat.view(n, self.k_size**2, n_head, d_v, h, w)

    
        #############################################################################################
        # TODO: Ver1. linear
        # # k 차원 축소
        # k = self.linear_k_reduce(k.permute(0, 2, 3, 4, 5, 1))  # [n, h, w, n_head, d_k, 392]
        # k = k.permute(0, 5, 1, 2, 3, 4)  # [n, 392, n_head, d_k, h, w]

        # # v 차원 축소
        # v = self.linear_v_reduce(v.permute(0, 2, 3, 4, 5, 1))  # [n, h, w, n_head, d_v, 392]
        # v = v.permute(0, 5, 1, 2, 3, 4)  # [n, 392, n_head, d_v, h, w]
        #############################################################################################
        # TODO: Ver2. conv -> 이거하면 그냥 ref를 그대로 복제한다. NCE를 안만 높여도. 이걸 줄이는게 왜 원본 복제를 하게할까
        # k = k.permute(0, 2, 3, 1, 4, 5).reshape(n * n_head * d_k, self.k_size**2, h, w)
        # v = v.permute(0, 2, 3, 1, 4, 5).reshape(n * n_head * d_k, self.k_size**2, h, w)

        # # Apply independent reductions
        # # n * nhead * dk, k^2/?, h, w
        # k = self.k_reduce_conv(k)
        # v = self.v_reduce_conv(v)

        # # Restore original batch structure
        # k = k.reshape(n, n_head, d_k, self.k_size**2 // 4, h, w).permute(0, 3, 1, 2, 4, 5)
        # v = v.reshape(n, n_head, d_k, self.k_size**2 // 4, h, w).permute(0, 3, 1, 2, 4, 5)
        #############################################################################################

        # -------------- Attention ----------------- (여기서 10GB 정도나 먹어) 나머지에서 이미 12GB를 먹는다는 얘기. 나머지에서도 메모리 소비가 이미 크다.
        if attn_type == "softmax":
            # n x 1 x nhead x dk x h x w --> n x nhead x dv x h x w
            q, attn = softmax_attention(q, k, v)
        elif attn_type == "dot":
            q, attn = dotproduct_attention(q, k, v)
        elif attn_type == "performer":
            q = performer_attention(q, k, v, performer_cross_attn=self.performer_cross_attn)
            attn = None
        elif attn_type == "flash":
            q = flash_attention(q, k, v)
            attn = None
        else:
            raise NotImplementedError(f"Unknown attention type {attn_type}")

        # Concatenate all the heads together
        # n x (nhead*dv) x h x w
        q = q.reshape(n, -1, h, w)
        q = q.float()
        q = self.fc(q)  # n x c x h x w

        return q, attn


def flow_to_grid(flow, k_size=5):
    # flow (Tensor): Tensor with size (n, 2, h, w), normal value.
    # samples = flow + grid + shift
    # n, h, w, _ = flow.size()
    n, _, h, w = flow.size()  # [4, 2, 48, 48]
    padding = (k_size - 1) // 2

    grid_y, grid_x = torch.meshgrid( # grid_x, grid_y = h, w
        torch.arange(0, h), torch.arange(0, w)
    )  # [48, 48] , [48, 48]
    grid_y = grid_y[None, ...].expand(k_size**2, -1, -1).type_as(flow) # [k^2, 48, 48] # 복제하여 grid를 확장
    grid_x = grid_x[None, ...].expand(k_size**2, -1, -1).type_as(flow) # [k^2, 48, 48]

    shift = torch.arange(0, k_size).type_as(flow) - padding  # [k] -> -2,-1,0,1,2 (if k=5)
    shift_y, shift_x = torch.meshgrid(shift, shift)  # [k, k]
    shift_y = shift_y.reshape(-1, 1, 1).expand(-1, h, w)  # k^2, h, w  # [25, 48, 48] # 첫번째 차원의 값에 따라 x,y방향으로 얼만큼씩 이동해야하는지. 일괄적으로 -2~2만큼 이동
    shift_x = shift_x.reshape(-1, 1, 1).expand(-1, h, w)  # k^2, h, w

    samples_y = grid_y + shift_y  # k^2, h, w  # [25, 48, 48]
    samples_x = grid_x + shift_x  # k^2, h, w
    # src에서 local grid를 생성.
    samples_grid = torch.stack(
        (samples_x, samples_y), 3
    )  # k^2, h, w, 2 # # [25, 48, 48, 2]
    samples_grid = samples_grid[None, ...].expand(
        n, -1, -1, -1, -1
    )  # n, k^2, h, w, 2  # [4, 25, 48, 48, 2]

    flow = flow.permute(0, 2, 3, 1)[:, None, ...].expand(
        -1, k_size**2, -1, -1, -1
    )  # [4, 25, 48, 48, 2]

    # ref에서의 local grid로 옮기는 작업.
    vgrid = samples_grid + flow  ## [4, 25, 48, 48, 2]  [4, 25, 48, 48, 2]
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[..., 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[..., 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=4).view(-1, h, w, 2)
    # vgrid_scaled.requires_grad = False
    return vgrid_scaled  # n x k^2, h, w, 2  # [100, 48, 48, 2]


def flow_guide_sampler(
    feat,
    vgrid_scaled,
    k_size=5,
    interp_mode="bilinear",
    padding_mode="zeros",
    align_corners=True,
):  # feat: [4, 64, 48, 48]
    # feat (Tensor): Tensor with size (n, c, h, w).
    # vgrid (Tensor): Tensor with size (nk^2, h, w, 2)
    n, c, h, w = feat.size()
    feat = (
        feat.view(n, 1, c, h, w).expand(-1, k_size**2, -1, -1, -1).reshape(-1, c, h, w)
    )  # (nk^2, c, h, w)
    sample_feat = F.grid_sample(
        feat,
        vgrid_scaled,  # 점을 이동시키는게 아니라 grid를 이동시키는거다. 그래서 src-> ref 로 얻은 flow를 ref의 grid에 적용해주는거구나. OK!!
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    ).view(
        n, k_size**2, c, h, w
    )  # [4, 25, 64, 48, 48]
    return sample_feat

