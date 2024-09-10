import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch.utils.data as Data
import nibabel as nib
import torch
import itertools


def generate_grid(imgshape):
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    z = np.arange(imgshape[2])
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


def generate_grid_unit(imgshape):
    x = (np.arange(imgshape[0]) - ((imgshape[0] - 1) / 2)) / (imgshape[0] - 1) * 2
    y = (np.arange(imgshape[1]) - ((imgshape[1] - 1) / 2)) / (imgshape[1] - 1) * 2
    z = (np.arange(imgshape[2]) - ((imgshape[2] - 1) / 2)) / (imgshape[2] - 1) * 2
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


def transform_unit_flow_to_flow(flow):
    x, y, z, _ = flow.shape
    flow[:, :, :, 0] = flow[:, :, :, 0] * (z-1)/2
    flow[:, :, :, 1] = flow[:, :, :, 1] * (y-1)/2
    flow[:, :, :, 2] = flow[:, :, :, 2] * (x-1)/2

    return flow


def transform_unit_flow_to_flow_cuda(flow):
    b, x, y, z, c = flow.shape
    flow[:, :, :, :, 0] = flow[:, :, :, :, 0] * (z-1)/2
    flow[:, :, :, :, 1] = flow[:, :, :, :, 1] * (y-1)/2
    flow[:, :, :, :, 2] = flow[:, :, :, :, 2] * (x-1)/2

    return flow


def load_4D(name):
    X = nib.load(name)
    X = X.get_fdata()
    X = np.reshape(X, (1,) + X.shape)
    return X


def load_5D(name):
    X = fixed_nii = nib.load(name)
    X = X.get_fdata()
    X = np.reshape(X, (1,) + (1,) + X.shape)
    return X


def imgnorm(img):
    max_v = np.max(img)
    min_v = np.min(img)

    norm_img = (img - min_v) / (max_v - min_v)
    return norm_img


def save_img(I_img,savename,header=None,affine=None):
    if header is None or affine is None:
        affine = np.diag([1, 1, 1, 1])
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    else:
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=header)

    nib.save(new_img, savename)


def save_img_nii(I_img, savename):
    affine = np.diag([1, 1, 1, 1])
    new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    nib.save(new_img, savename)


def save_flow(I_img,savename,header=None,affine=None):
    if header is None or affine is None:
        affine = np.diag([1, 1, 1, 1])
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    else:
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=header)

    nib.save(new_img, savename)


class Dataset(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, names, iterations, norm=False):
        'Initialization'
        self.names = names
        self.norm = norm
        self.iterations = iterations

    def __len__(self):
        'Denotes the total number of samples'
        return self.iterations

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        index_pair = np.random.permutation(len(self.names))[0:2]
        img_A = load_4D(self.names[index_pair[0]])
        img_B = load_4D(self.names[index_pair[1]])
        if self.norm:
            return torch.from_numpy(imgnorm(img_A)).float(), torch.from_numpy(imgnorm(img_B)).float()
        else:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()


class Dataset_epoch(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, names, norm=False):
        'Initialization'
        self.names = names
        self.norm = norm
        self.index_pair = list(itertools.permutations(names, 2))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        img_A = load_4D(self.index_pair[step][0])
        img_B = load_4D(self.index_pair[step][1])

        if self.norm:
            return torch.from_numpy(imgnorm(img_A)).float(), torch.from_numpy(imgnorm(img_B)).float()
        else:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()


class Dataset_epoch_validation(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, imgs, labels, norm=False):
        'Initialization'
        super(Dataset_epoch_validation, self).__init__()

        self.imgs = imgs
        self.labels = labels
        self.norm = norm
        self.imgs_pair = list(itertools.permutations(imgs, 2))
        self.labels_pair = list(itertools.permutations(labels, 2))

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.imgs_pair)

  def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        img_A = load_4D(self.imgs_pair[step][0])
        img_B = load_4D(self.imgs_pair[step][1])

        label_A = load_4D(self.labels_pair[step][0])
        label_B = load_4D(self.labels_pair[step][1])

        # print(self.index_pair[step][0])
        # print(self.index_pair[step][1])

        if self.norm:
            return torch.from_numpy(imgnorm(img_A)).float(), torch.from_numpy(imgnorm(img_B)).float(), torch.from_numpy(label_A).float(), torch.from_numpy(label_B).float()
        else:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float(), torch.from_numpy(label_A).float(), torch.from_numpy(label_B).float()


class Predict_dataset(Data.Dataset):
    def __init__(self, fixed_list, move_list, fixed_label_list, move_label_list, norm=False):
        super(Predict_dataset, self).__init__()
        self.fixed_list = fixed_list
        self.move_list = move_list
        self.fixed_label_list = fixed_label_list
        self.move_label_list = move_label_list
        self.norm = norm

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.move_list)

    def __getitem__(self, index):
        fixed_img = load_4D(self.fixed_list)
        moved_img = load_4D(self.move_list[index])
        fixed_label = load_4D(self.fixed_label_list)
        moved_label = load_4D(self.move_label_list[index])

        if self.norm:
            fixed_img = imgnorm(fixed_img)
            moved_img = imgnorm(moved_img)

        fixed_img = torch.from_numpy(fixed_img)
        moved_img = torch.from_numpy(moved_img)
        fixed_label = torch.from_numpy(fixed_label)
        moved_label = torch.from_numpy(moved_label)

        if self.norm:
            output = {'fixed': fixed_img.float(), 'move': moved_img.float(),
                      'fixed_label': fixed_label.float(), 'move_label': moved_label.float(), 'index': index}
            return output
        else:
            output = {'fixed': fixed_img.float(), 'move': moved_img.float(),
                      'fixed_label': fixed_label.float(), 'move_label': moved_label.float(), 'index': index}
            return output



class Miccai2020_LDR_laplacian_unit_add_lvl1(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, imgshape=(160, 192, 144), range_flow=0.4):
        super(Miccai2020_LDR_laplacian_unit_add_lvl1, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.imgshape = imgshape

        self.grid_1 = generate_grid_unit(self.imgshape)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).cuda().float()

        self.diff_transform = DiffeomorphicTransform_unit(time_step=7).cuda()
        self.transform = SpatialTransform_unit().cuda()

        bias_opt = False

        self.input_encoder_lvl1 = self.input_feature_extract(self.in_channel, self.start_channel * 4, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1, bias=bias_opt)

        self.resblock_group_lvl1 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                               padding=0, output_padding=0, bias=bias_opt)

        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        self.output_lvl1 = self.outputs(self.start_channel * 8, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)


    def resblock_seq(self, in_channels, bias_opt=False):
        layer = nn.Sequential(
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2)
        )
        return layer


    def input_feature_extract(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, int(in_channels/2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(int(in_channels/2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, x, y):

        cat_input = torch.cat((x, y), 1)
        cat_input = self.down_avg(cat_input)
        cat_input_lvl1 = self.down_avg(cat_input)

        down_y = cat_input_lvl1[:, 1:2, :, :, :]

        fea_e0 = self.input_encoder_lvl1(cat_input_lvl1)
        e0 = self.down_conv(fea_e0)
        e0 = self.resblock_group_lvl1(e0)
        e0 = self.up(e0)
        output_disp_e0_v = self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        output_disp_e0 = self.diff_transform(output_disp_e0_v, self.grid_1)
        warpped_inputx_lvl1_out = self.transform(x, output_disp_e0.permute(0, 2, 3, 4, 1), self.grid_1)


        if self.is_train is True:
            return output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
        else:
            return output_disp_e0


class Miccai2020_LDR_laplacian_unit_add_lvl2(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, imgshape=(160, 192, 144), range_flow=0.4, model_lvl1=None):
        super(Miccai2020_LDR_laplacian_unit_add_lvl2, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.imgshape = imgshape

        self.model_lvl1 = model_lvl1

        self.grid_1 = generate_grid_unit(self.imgshape)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).cuda().float()

        self.diff_transform = DiffeomorphicTransform_unit(time_step=7).cuda()
        self.transform = SpatialTransform_unit().cuda()

        bias_opt = False

        self.input_encoder_lvl1 = self.input_feature_extract(self.in_channel+3, self.start_channel * 4, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1, bias=bias_opt)

        self.resblock_group_lvl1 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear")
        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        self.output_lvl1 = self.outputs(self.start_channel * 8, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)

    def unfreeze_modellvl1(self):
        # unFreeze model_lvl1 weight
        print("\nunfreeze model_lvl1 parameter")
        for param in self.model_lvl1.parameters():
            param.requires_grad = True

    def resblock_seq(self, in_channels, bias_opt=False):
        layer = nn.Sequential(
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2)
        )
        return layer

    def input_feature_extract(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                              bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, int(in_channels/2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(int(in_channels/2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, x, y):
        lvl1_disp, _, _, lvl1_v, lvl1_embedding = self.model_lvl1(x, y)
        lvl1_disp_up = self.up_tri(lvl1_disp)
        lvl1_v_up = self.up_tri(lvl1_v)

        x_down = self.down_avg(x)
        y_down = self.down_avg(y)

        warpped_x = self.transform(x_down, lvl1_disp_up.permute(0, 2, 3, 4, 1), self.grid_1)

        cat_input_lvl2 = torch.cat((warpped_x, y_down, lvl1_v_up), 1)

        fea_e0 = self.input_encoder_lvl1(cat_input_lvl2)
        e0 = self.down_conv(fea_e0)

        e0 = e0 + lvl1_embedding

        e0 = self.resblock_group_lvl1(e0)
        e0 = self.up(e0)
        output_disp_e0_v = self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        compose_field_e0_lvl1v = output_disp_e0_v + lvl1_v_up
        output_disp_e0 = self.diff_transform(compose_field_e0_lvl1v, self.grid_1)
        warpped_inputx_lvl1_out = self.transform(x, output_disp_e0.permute(0, 2, 3, 4, 1), self.grid_1)

        if self.is_train is True:
            return output_disp_e0, warpped_inputx_lvl1_out, y_down, compose_field_e0_lvl1v, lvl1_v, e0
        else:
            return output_disp_e0


class Miccai2020_LDR_laplacian_unit_add_lvl3(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, imgshape=(160, 192, 144), range_flow=0.4,
                 model_lvl2=None):
        super(Miccai2020_LDR_laplacian_unit_add_lvl3, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.imgshape = imgshape

        self.model_lvl2 = model_lvl2

        self.grid_1 = generate_grid_unit(self.imgshape)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).cuda().float()

        self.diff_transform = DiffeomorphicTransform_unit(time_step=7).cuda()
        self.transform = SpatialTransform_unit().cuda()

        bias_opt = False

        self.input_encoder_lvl1 = self.input_feature_extract(self.in_channel+3, self.start_channel * 4, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1, bias=bias_opt)

        self.resblock_group_lvl1 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear")
        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        self.output_lvl1 = self.outputs(self.start_channel * 8, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)


    def unfreeze_modellvl2(self):
        # unFreeze model_lvl1 weight
        print("\nunfreeze model_lvl2 parameter")
        for param in self.model_lvl2.parameters():
            param.requires_grad = True

    def resblock_seq(self, in_channels, bias_opt=False):
        layer = nn.Sequential(
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2)
        )
        return layer

    def input_feature_extract(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                              bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, int(in_channels/2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(int(in_channels/2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, x, y):
        # output_disp_e0, warpped_inputx_lvl1_out, y_down, compose_field_e0_lvl1v, lvl1_v, e0
        lvl2_disp, _, _, compose_lvl2_v, lvl1_v, lvl2_embedding = self.model_lvl2(x, y)
        lvl2_disp_up = self.up_tri(lvl2_disp)
        compose_lvl2_v_up = self.up_tri(compose_lvl2_v)
        warpped_x = self.transform(x, lvl2_disp_up.permute(0, 2, 3, 4, 1), self.grid_1)

        cat_input = torch.cat((warpped_x, y, compose_lvl2_v_up), 1)

        fea_e0 = self.input_encoder_lvl1(cat_input)
        e0 = self.down_conv(fea_e0)

        e0 = e0 + lvl2_embedding
        e0 = self.resblock_group_lvl1(e0)
        e0 = self.up(e0)
        output_disp_e0_v = self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        compose_field_e0_lvl2_compose = output_disp_e0_v + compose_lvl2_v_up
        output_disp_e0 = self.diff_transform(compose_field_e0_lvl2_compose, self.grid_1)

        warpped_inputx_lvl1_out = self.transform(x, output_disp_e0.permute(0, 2, 3, 4, 1), self.grid_1)

        if self.is_train is True:
            return output_disp_e0, warpped_inputx_lvl1_out, y, compose_field_e0_lvl2_compose, lvl1_v, compose_lvl2_v, e0
        else:
            return output_disp_e0


class Miccai2020_LDR_laplacian_unit_disp_add_lvl1(nn.Module):
    # def __init__(self, in_channel, n_classes, start_channel, is_train=True, imgshape=(160, 192, 144), range_flow=0.4):
    def __init__(self, **kwargs):
        try:
            in_channel = kwargs['in_channel']
            n_classes = kwargs['n_classes']
            start_channel = kwargs['start_channel']
            is_train = kwargs['is_train']
            imgshape = kwargs['imgshape']
            range_flow = kwargs['range_flow']
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {str(e)}")
        

        super(Miccai2020_LDR_laplacian_unit_disp_add_lvl1, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.imgshape = imgshape

        self.grid_1 = generate_grid_unit(self.imgshape)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).cuda().float()

        self.transform = SpatialTransform_unit().cuda()

        bias_opt = False

        self.input_encoder_lvl1 = self.input_feature_extract(self.in_channel, self.start_channel * 4, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1, bias=bias_opt)

        self.resblock_group_lvl1 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                               padding=0, output_padding=0, bias=bias_opt)

        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        self.output_lvl1 = self.outputs(self.start_channel * 8, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)


    def resblock_seq(self, in_channels, bias_opt=False):
        layer = nn.Sequential(
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2)
        )
        return layer


    def input_feature_extract(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, int(in_channels/2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(int(in_channels/2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, x, y):

        cat_input = torch.cat((x, y), 1)
        cat_input = self.down_avg(cat_input)
        cat_input_lvl1 = self.down_avg(cat_input)

        down_y = cat_input_lvl1[:, 1:2, :, :, :]

        fea_e0 = self.input_encoder_lvl1(cat_input_lvl1)
        e0 = self.down_conv(fea_e0)
        e0 = self.resblock_group_lvl1(e0)
        e0 = self.up(e0)
        output_disp_e0_v = self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        warpped_inputx_lvl1_out = self.transform(x, output_disp_e0_v.permute(0, 2, 3, 4, 1), self.grid_1)


        if self.is_train is True:
            return output_disp_e0_v, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
        else:
            return output_disp_e0_v


class Miccai2020_LDR_laplacian_unit_disp_add_lvl2(nn.Module):
    # def __init__(self, in_channel, n_classes, start_channel, is_train=True, imgshape=(160, 192, 144), range_flow=0.4, model_lvl1=None):
    def __init__(self, **kwargs):
        try:
            in_channel = kwargs['in_channel']
            n_classes = kwargs['n_classes']
            start_channel = kwargs['start_channel']
            is_train = kwargs['is_train']
            imgshape = kwargs['imgshape']
            range_flow = kwargs['range_flow']
            model_lvl1 = kwargs['model_lvl1'] # path -> model
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {str(e)}")
        
        super(Miccai2020_LDR_laplacian_unit_disp_add_lvl2, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.imgshape = imgshape

        imgshape_lvl1 = (imgshape[0] // 2, imgshape[1] // 2, imgshape[2] // 2)

        kwargs_lvl1 = kwargs.copy()
        kwargs_lvl1['imgshape'] = imgshape_lvl1

        # checkpoint = torch.load(model_lvl1, map_location=lambda storage, loc: storage)
        # model_state_dict = checkpoint["state_dict"]
        # adjusted_state_dict = {k.replace("netR_1.", ""): v for k, v in model_state_dict.items()}
        self.model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(**kwargs_lvl1)
        # self.model_lvl1.load_state_dict(adjusted_state_dict, strict=False)
        
        # for param in self.model_lvl1.parameters():
        #     param.requires_grad = False

        self.grid_1 = generate_grid_unit(self.imgshape)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).cuda().float()

        self.transform = SpatialTransform_unit().cuda()

        bias_opt = False

        self.input_encoder_lvl1 = self.input_feature_extract(self.in_channel+3, self.start_channel * 4, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1, bias=bias_opt)

        self.resblock_group_lvl1 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear")
        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        self.output_lvl1 = self.outputs(self.start_channel * 8, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)

    def unfreeze_modellvl1(self):
        # unFreeze model_lvl1 weight
        print("\nunfreeze model_lvl1 parameter")
        for param in self.model_lvl1.parameters():
            param.requires_grad = True

    def resblock_seq(self, in_channels, bias_opt=False):
        layer = nn.Sequential(
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2)
        )
        return layer

    def input_feature_extract(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                              bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, int(in_channels/2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(int(in_channels/2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, x, y):
        # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
        lvl1_disp, _, _, lvl1_v, lvl1_embedding = self.model_lvl1(x, y)
        lvl1_disp_up = self.up_tri(lvl1_disp)

        x_down = self.down_avg(x)
        y_down = self.down_avg(y)

        warpped_x = self.transform(x_down, lvl1_disp_up.permute(0, 2, 3, 4, 1), self.grid_1)

        cat_input_lvl2 = torch.cat((warpped_x, y_down, lvl1_disp_up), 1)

        fea_e0 = self.input_encoder_lvl1(cat_input_lvl2)
        e0 = self.down_conv(fea_e0)

        e0 = e0 + lvl1_embedding

        e0 = self.resblock_group_lvl1(e0)
        e0 = self.up(e0)
        output_disp_e0_v = self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        compose_field_e0_lvl1 = lvl1_disp_up + output_disp_e0_v
        warpped_inputx_lvl1_out = self.transform(x, compose_field_e0_lvl1.permute(0, 2, 3, 4, 1), self.grid_1)

        if self.is_train is True:
            return compose_field_e0_lvl1, warpped_inputx_lvl1_out, y_down, output_disp_e0_v, lvl1_v, e0
        else:
            return compose_field_e0_lvl1


class Miccai2020_LDR_laplacian_unit_disp_add_lvl3(nn.Module):
    # def __init__(self, in_channel, n_classes, start_channel, is_train=True, imgshape=(160, 192, 144), range_flow=0.4, model_lvl2=None):
    def __init__(self, **kwargs):
        try:
            in_channel = kwargs['in_channel']
            n_classes = kwargs['n_classes']
            start_channel = kwargs['start_channel']
            is_train = kwargs['is_train']
            imgshape = kwargs['imgshape']
            range_flow = kwargs['range_flow']
            model_lvl1 = kwargs['model_lvl1']
            model_lvl2 = kwargs['model_lvl2']
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {str(e)}")
        
        super(Miccai2020_LDR_laplacian_unit_disp_add_lvl3, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.imgshape = imgshape

        # imgshape_lvl1 = (imgshape[0] // 4, imgshape[1] // 4, imgshape[2] // 4)
        imgshape_lvl2 = (imgshape[0] // 2, imgshape[1] // 2, imgshape[2] // 2)

        # kwargs_lvl1 = kwargs.copy()
        # kwargs_lvl1['imgshape'] = imgshape_lvl1

        # checkpoint = torch.load(model_lvl1, map_location=lambda storage, loc: storage)
        # model_state_dict = checkpoint["state_dict"]
        # adjusted_state_dict = {k.replace("netR_1.", ""): v for k, v in model_state_dict.items()}
        # self.model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(**kwargs_lvl1)
        # self.model_lvl1.load_state_dict(adjusted_state_dict, strict=False)
        
        kwargs_lvl2 = kwargs.copy()
        kwargs_lvl2['imgshape'] = imgshape_lvl2
        kwargs_lvl2['model_lvl1'] = model_lvl1

        # checkpoint = torch.load(model_lvl2, map_location=lambda storage, loc: storage)
        # model_state_dict = checkpoint["state_dict"]
        # adjusted_state_dict = {k.replace("netR_2.", ""): v for k, v in model_state_dict.items()}
        self.model_lvl2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(**kwargs_lvl2)
        # self.model_lvl2.load_state_dict(adjusted_state_dict, strict=False)

        self.grid_1 = generate_grid_unit(self.imgshape)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).cuda().float()

        self.transform = SpatialTransform_unit().cuda()

        bias_opt = False

        self.input_encoder_lvl1 = self.input_feature_extract(self.in_channel+3, self.start_channel * 4, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1, bias=bias_opt)

        self.resblock_group_lvl1 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear")
        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        self.output_lvl1 = self.outputs(self.start_channel * 8, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)


    def unfreeze_modellvl2(self):
        # unFreeze model_lvl1 weight
        print("\nunfreeze model_lvl2 parameter")
        for param in self.model_lvl2.parameters():
            param.requires_grad = True

    def resblock_seq(self, in_channels, bias_opt=False):
        layer = nn.Sequential(
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2)
        )
        return layer

    def input_feature_extract(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                              bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, int(in_channels/2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(int(in_channels/2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, x, y):
        # compose_field_e0_lvl1, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, lvl1_v, e0
        lvl2_disp, _, _, lvl2_v, lvl1_v, lvl2_embedding = self.model_lvl2(x, y)
        lvl2_disp_up = self.up_tri(lvl2_disp)
        warpped_x = self.transform(x, lvl2_disp_up.permute(0, 2, 3, 4, 1), self.grid_1)

        cat_input = torch.cat((warpped_x, y, lvl2_disp_up), 1)

        fea_e0 = self.input_encoder_lvl1(cat_input)
        e0 = self.down_conv(fea_e0)

        e0 = e0 + lvl2_embedding
        e0 = self.resblock_group_lvl1(e0)
        e0 = self.up(e0)
        output_disp_e0_v = self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        compose_field_e0_lvl1 = output_disp_e0_v + lvl2_disp_up

        warpped_inputx_lvl1_out = self.transform(x, compose_field_e0_lvl1.permute(0, 2, 3, 4, 1), self.grid_1)

        if self.is_train is True:
            return compose_field_e0_lvl1, warpped_inputx_lvl1_out, y, output_disp_e0_v, lvl1_v, lvl2_v, e0
        else:
            return compose_field_e0_lvl1


class Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl1(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, imgshape=(160, 192, 144), range_flow=0.4):
        super(Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl1, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.imgshape = imgshape

        self.grid_1 = generate_grid(self.imgshape)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).cuda().float()

        self.transform = SpatialTransform().cuda()

        bias_opt = False

        self.input_encoder_lvl1 = self.input_feature_extract(self.in_channel, self.start_channel * 4, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1, bias=bias_opt)

        self.resblock_group_lvl1 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                               padding=0, output_padding=0, bias=bias_opt)

        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        self.output_lvl1 = self.outputs(self.start_channel * 8, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)


    def resblock_seq(self, in_channels, bias_opt=False):
        layer = nn.Sequential(
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2)
        )
        return layer


    def input_feature_extract(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, int(in_channels/2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(int(in_channels/2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, x, y):

        cat_input = torch.cat((x, y), 1)
        cat_input = self.down_avg(cat_input)
        cat_input_lvl1 = self.down_avg(cat_input)

        start_s, end_s = self.in_channel//2, self.in_channel
        down_y = cat_input_lvl1[:, start_s:end_s, :, :, :]

        fea_e0 = self.input_encoder_lvl1(cat_input_lvl1)
        e0 = self.down_conv(fea_e0)
        e0 = self.resblock_group_lvl1(e0)
        e0 = self.up(e0)
        output_disp_e0_v = self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        warpped_inputx_lvl1_out = self.transform(x, output_disp_e0_v.permute(0, 2, 3, 4, 1), self.grid_1)


        if self.is_train is True:
            return output_disp_e0_v, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
        else:
            return output_disp_e0_v


class Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl2(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, imgshape=(160, 192, 144), range_flow=0.4, model_lvl1=None):
        super(Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl2, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.imgshape = imgshape

        self.model_lvl1 = model_lvl1

        self.grid_1 = generate_grid(self.imgshape)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).cuda().float()

        self.transform = SpatialTransform().cuda()

        bias_opt = False

        self.input_encoder_lvl1 = self.input_feature_extract(self.in_channel+3, self.start_channel * 4, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1, bias=bias_opt)

        self.resblock_group_lvl1 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear")
        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        self.output_lvl1 = self.outputs(self.start_channel * 8, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)

    def unfreeze_modellvl1(self):
        # unFreeze model_lvl1 weight
        print("\nunfreeze model_lvl1 parameter")
        for param in self.model_lvl1.parameters():
            param.requires_grad = True

    def resblock_seq(self, in_channels, bias_opt=False):
        layer = nn.Sequential(
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2)
        )
        return layer

    def input_feature_extract(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                              bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, int(in_channels/2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(int(in_channels/2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, x, y):
        # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
        lvl1_disp, _, _, lvl1_v, lvl1_embedding = self.model_lvl1(x, y)
        lvl1_disp_up = self.up_tri(lvl1_disp) * 2.

        x_down = self.down_avg(x)
        y_down = self.down_avg(y)

        warpped_x = self.transform(x_down, lvl1_disp_up.permute(0, 2, 3, 4, 1), self.grid_1)

        cat_input_lvl2 = torch.cat((warpped_x, y_down, lvl1_disp_up), 1)

        fea_e0 = self.input_encoder_lvl1(cat_input_lvl2)
        e0 = self.down_conv(fea_e0)

        e0 = e0 + lvl1_embedding

        e0 = self.resblock_group_lvl1(e0)
        e0 = self.up(e0)
        output_disp_e0_v = self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        compose_field_e0_lvl1 = lvl1_disp_up + output_disp_e0_v
        warpped_inputx_lvl1_out = self.transform(x, compose_field_e0_lvl1.permute(0, 2, 3, 4, 1), self.grid_1)

        if self.is_train is True:
            return compose_field_e0_lvl1, warpped_inputx_lvl1_out, y_down, output_disp_e0_v, lvl1_v, e0
        else:
            return compose_field_e0_lvl1


class Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl3(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, imgshape=(160, 192, 144), range_flow=0.4,
                 model_lvl2=None):
        super(Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl3, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.imgshape = imgshape

        self.model_lvl2 = model_lvl2

        self.grid_1 = generate_grid(self.imgshape)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).cuda().float()

        self.transform = SpatialTransform().cuda()

        bias_opt = False

        self.input_encoder_lvl1 = self.input_feature_extract(self.in_channel+3, self.start_channel * 4, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1, bias=bias_opt)

        self.resblock_group_lvl1 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear")
        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        self.output_lvl1 = self.outputs(self.start_channel * 8, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)


    def unfreeze_modellvl2(self):
        # unFreeze model_lvl1 weight
        print("\nunfreeze model_lvl2 parameter")
        for param in self.model_lvl2.parameters():
            param.requires_grad = True

    def resblock_seq(self, in_channels, bias_opt=False):
        layer = nn.Sequential(
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2)
        )
        return layer

    def input_feature_extract(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                              bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, int(in_channels/2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(int(in_channels/2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, x, y):
        # compose_field_e0_lvl1, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, lvl1_v, e0
        lvl2_disp, _, _, lvl2_v, lvl1_v, lvl2_embedding = self.model_lvl2(x, y)
        lvl2_disp_up = self.up_tri(lvl2_disp) * 2.
        warpped_x = self.transform(x, lvl2_disp_up.permute(0, 2, 3, 4, 1), self.grid_1)

        cat_input = torch.cat((warpped_x, y, lvl2_disp_up), 1)

        fea_e0 = self.input_encoder_lvl1(cat_input)
        e0 = self.down_conv(fea_e0)

        e0 = e0 + lvl2_embedding
        e0 = self.resblock_group_lvl1(e0)
        e0 = self.up(e0)
        output_disp_e0_v = self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        compose_field_e0_lvl1 = output_disp_e0_v + lvl2_disp_up

        warpped_inputx_lvl1_out = self.transform(x, compose_field_e0_lvl1.permute(0, 2, 3, 4, 1), self.grid_1)

        if self.is_train is True:
            return compose_field_e0_lvl1, warpped_inputx_lvl1_out, y, output_disp_e0_v, lvl1_v, lvl2_v, e0
        else:
            return compose_field_e0_lvl1


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, num_group=4, stride=1, bias=False):
        super(PreActBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=bias)
            )


    def forward(self, x):
        out = F.leaky_relu(x, negative_slope=0.2)

        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)

        out = self.conv2(F.leaky_relu(out, negative_slope=0.2))


        out += shortcut
        return out


class SpatialTransform_unit(nn.Module):
    def __init__(self):
        super(SpatialTransform_unit, self).__init__()

    def forward(self, x, flow, sample_grid):
        sample_grid = sample_grid + flow
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='bilinear', padding_mode="border", align_corners=True)
        return flow


class SpatialTransformNearest_unit(nn.Module):
    def __init__(self):
        super(SpatialTransformNearest_unit, self).__init__()

    def forward(self, x, flow, sample_grid):
        sample_grid = sample_grid + flow
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='nearest', padding_mode="border", align_corners=True)
        return flow


class DiffeomorphicTransform_unit(nn.Module):
    def __init__(self, time_step=7):
        super(DiffeomorphicTransform_unit, self).__init__()
        self.time_step = time_step

    def forward(self, velocity, sample_grid):
        flow = velocity/(2.0**self.time_step)
        for _ in range(self.time_step):
            grid = sample_grid + flow.permute(0,2,3,4,1)
            flow = flow + F.grid_sample(flow, grid, mode='bilinear', padding_mode="border", align_corners=True)
        return flow


class SpatialTransform(nn.Module):
    def __init__(self):
        super(SpatialTransform, self).__init__()

    def forward(self, x, flow, sample_grid):
        sample_grid = sample_grid + flow
        size_tensor = sample_grid.size()
        sample_grid[0, :, :, :, 0] = (sample_grid[0, :, :, :, 0] - ((size_tensor[3] - 1) / 2)) / (
                size_tensor[3] - 1) * 2
        sample_grid[0, :, :, :, 1] = (sample_grid[0, :, :, :, 1] - ((size_tensor[2] - 1) / 2)) / (
                size_tensor[2] - 1) * 2
        sample_grid[0, :, :, :, 2] = (sample_grid[0, :, :, :, 2] - ((size_tensor[1] - 1) / 2)) / (
                size_tensor[1] - 1) * 2
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='bilinear', align_corners=True)

        return flow


class SpatialTransformNearest(nn.Module):
    def __init__(self):
        super(SpatialTransformNearest, self).__init__()

    def forward(self, x, flow, sample_grid):
        sample_grid = sample_grid + flow
        size_tensor = sample_grid.size()
        sample_grid[0, :, :, :, 0] = (sample_grid[0, :, :, :, 0] - ((size_tensor[3] - 1) / 2)) / (
                    size_tensor[3] - 1) * 2
        sample_grid[0, :, :, :, 1] = (sample_grid[0, :, :, :, 1] - ((size_tensor[2] - 1) / 2)) / (
                    size_tensor[2] - 1) * 2
        sample_grid[0, :, :, :, 2] = (sample_grid[0, :, :, :, 2] - ((size_tensor[1] - 1) / 2)) / (
                    size_tensor[1] - 1) * 2
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='nearest', align_corners=True)

        return flow


class DiffeomorphicTransform(nn.Module):
    def __init__(self, time_step=7):
        super(DiffeomorphicTransform, self).__init__()
        self.time_step = time_step

    def forward(self, velocity, sample_grid, range_flow):
        flow = velocity / (2.0 ** self.time_step)
        size_tensor = sample_grid.size()
        # 0.5 flow
        for _ in range(self.time_step):
            grid = sample_grid + (flow.permute(0, 2, 3, 4, 1) * range_flow)
            grid[0, :, :, :, 0] = (grid[0, :, :, :, 0] - ((size_tensor[3] - 1) / 2)) / (size_tensor[3] - 1) * 2
            grid[0, :, :, :, 1] = (grid[0, :, :, :, 1] - ((size_tensor[2] - 1) / 2)) / (size_tensor[2] - 1) * 2
            grid[0, :, :, :, 2] = (grid[0, :, :, :, 2] - ((size_tensor[1] - 1) / 2)) / (size_tensor[1] - 1) * 2
            flow = flow + F.grid_sample(flow, grid, mode='bilinear', align_corners=True)
        return flow


def smoothloss(y_pred):
    dy = torch.abs(y_pred[:,:,1:, :, :] - y_pred[:,:, :-1, :, :])
    dx = torch.abs(y_pred[:,:,:, 1:, :] - y_pred[:,:, :, :-1, :])
    dz = torch.abs(y_pred[:,:,:, :, 1:] - y_pred[:,:, :, :, :-1])
    return (torch.mean(dx * dx)+torch.mean(dy*dy)+torch.mean(dz*dz))/3.0


def JacboianDet(y_pred, sample_grid):
    J = y_pred + sample_grid
    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    Jdet0 = dx[:,:,:,:,0] * (dy[:,:,:,:,1] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,1])
    Jdet1 = dx[:,:,:,:,1] * (dy[:,:,:,:,0] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,0])
    Jdet2 = dx[:,:,:,:,2] * (dy[:,:,:,:,0] * dz[:,:,:,:,1] - dy[:,:,:,:,1] * dz[:,:,:,:,0])

    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet


def neg_Jdet_loss(y_pred, sample_grid):
    neg_Jdet = -1.0 * JacboianDet(y_pred, sample_grid)
    selected_neg_Jdet = F.relu(neg_Jdet)

    return torch.mean(selected_neg_Jdet)


class NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=5, eps=1e-5):
        super(NCC, self).__init__()
        self.win = win
        self.eps = eps
        self.w_temp = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.w_temp

        # set window size
        if self.win is None:
            self.win = [5] * ndims
        else:
            self.win = [self.w_temp] * ndims

        weight_win_size = self.w_temp
        weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2))
        J_sum = conv_fn(J, weight, padding=int(win_size/2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)


class multi_resolution_NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def  __init__(self, win=None, eps=1e-5, scale=3):
        super(multi_resolution_NCC, self).__init__()
        self.num_scale = scale
        self.similarity_metric = []

        for i in range(scale):
            self.similarity_metric.append(NCC(win=win - (i*2)))

    def forward(self, I, J):
        total_NCC = []

        for i in range(self.num_scale):
            current_NCC = self.similarity_metric[i](I, J)
            total_NCC.append(current_NCC/(2**i))

            I = nn.functional.avg_pool3d(I, kernel_size=3, stride=2, padding=1, count_include_pad=False)
            J = nn.functional.avg_pool3d(J, kernel_size=3, stride=2, padding=1, count_include_pad=False)

        return sum(total_NCC)
