import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as io
# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import numpy as np
import torchvision
import cv2
from utils.loss_util import *
from utils.common import *
from torch.nn.parameter import Parameter
import math
from torch.autograd import Variable
# from model import MainNet, PcdAlign
from torch.autograd import Function
from complexPyTorch import complexFunctions, complexLayers
from complexPyTorch.complexLayers import ComplexBatchNorm2d,\
    ComplexLinear, ComplexReLU, ComplexMaxPool2d, apply_complex
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint
from model import MainNet, PcdAlign, pytorch_ssim
from .unet_parts import *


class AttentionModule(nn.Module):
    def __init__(self, channel):
        super(AttentionModule, self).__init__()
        self.conv_0 = nn.Sequential(
            nn.Conv2d(channel, channel, 1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv_0(x)
        z = self.conv_du(y)
        return z * y + x


class SPPF(nn.Module):
    def __init__(self, c_in, c_out):
        super(SPPF, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_in, 3, 1, 1)
        self.c1 = nn.MaxPool2d(5, 1, 2)
        self.c2 = nn.MaxPool2d(5, 1, 2)
        self.c3 = nn.MaxPool2d(5, 1, 2)
        self.conv2 = nn.Conv2d(c_in*4, c_out, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        out1 = x
        out2 = self.c1(x)
        out3 = self.c2(out2)
        out4 = self.c3(out3)
        out = torch.cat((out1, out2, out3, out4), 1)
        out = self.conv2(out)

        return out


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=20, bias=False):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5


class AttCouplingBlock(nn.Module):
    def __init__(self, dims_in, clamp=1.0, split=3):
        super().__init__()

        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])
        self.split_len1 = split
        self.split_len2 = channels - split
        self.split = split

        self.clamp = clamp
        self.affine_eps = 0.0001

        self.s1 = DenseBlock(self.split, self.split * 2, gc=19)
        self.t1 = DenseBlock(self.split, self.split * 2, gc=19)
        self.s2 = DenseBlock(self.split * 2, self.split)
        self.t2 = DenseBlock(self.split * 2, self.split)

        self.att_t1 = AttentionModule(self.split*2)
        self.att_t2 = AttentionModule(self.split)
        self.att_s1 = AttentionModule(self.split*2)
        self.att_s2 = AttentionModule(self.split)

    def e(self, s):
        return torch.exp(self.clamp * (torch.sigmoid(s) * 2 - 1)) + self.affine_eps

    def forward(self, x, rev=False):
        if not rev:
            x1, x2 = x[:, 0: self.split, :, :], x[:, self.split:, :, :]
            s1, t1 = self.s1(x1), self.t1(x1)
            s1, t1 = self.att_s1(s1), self.att_t1(t1)
            y2 = self.e(s1) * x2 + t1

            s2, t2 = self.s2(y2), self.t2(y2)
            s2, t2 = self.att_s2(s2), self.att_t2(t2)
            y1 = self.e(s2) * x1 + t2

            out = torch.cat((y1, y2), 1)
            return out
        else:
            y1, y2 = x[:, 0: self.split, :, :], x[:, self.split:, :, :]

            s2, t2 = self.s2(y2), self.t2(y2)
            s2, t2 = self.att_s2(s2), self.att_t2(t2)
            x1 = (y1 - t2) / self.e(s2)

            s1, t1 = self.s1(x1), self.t1(x1)
            s1, t1 = self.att_s1(s1), self.att_t1(t1)
            x2 = (y2 - t1) / self.e(s1)

            out = torch.cat((x1, x2), 1)
            return out


class RNVPCouplingBlock(nn.Module):
    def __init__(self, dims_in, clamp=1.0, split=3):
        super().__init__()

        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])
        self.split_len1 = split
        self.split_len2 = channels - split
        self.split = split

        self.clamp = clamp
        self.affine_eps = 0.0001

        self.s1 = DenseBlock(self.split, self.split*2, gc=19)
        self.t1 = DenseBlock(self.split, self.split*2, gc=19)
        # self.s1f = DenseBlock(self.split*2, self.split*2)
        # self.t1f = DenseBlock(self.split*2, self.split*2)
        self.s2 = DenseBlock(self.split * 2, self.split)
        self.t2 = DenseBlock(self.split * 2, self.split)

    def e(self, s):
        return torch.exp(self.clamp * (torch.sigmoid(s) * 2 - 1)) + self.affine_eps

    def forward(self, x, rev=False):
        if not rev:
            x1, x2 = x[:, 0: self.split, :, :], x[:, self.split:, :, :]
            s1, t1 = self.s1(x1), self.t1(x1)
            y2 = self.e(s1) * x2 + t1
            s2, t2 = self.s2(y2), self.t2(y2)
            y1 = self.e(s2) * x1 + t2

            out = torch.cat((y1, y2), 1)
            return out
        else:
            y1, y2 = x[:, 0: self.split, :, :], x[:, self.split:, :, :]

            s2, t2 = self.s2(y2), self.t2(y2)
            x1 = (y1 - t2) / self.e(s2)

            s1, t1 = self.s1(x1), self.t1(x1)
            x2 = (y2 - t1) / self.e(s1)

            out = torch.cat((x1, x2), 1)
            return out


class SPPFCB(nn.Module):
    def __init__(self, dims_in, clamp=1.0, split=3):
        super().__init__()

        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])
        self.split_len1 = split
        self.split_len2 = channels - split
        self.split = split

        self.clamp = clamp
        self.affine_eps = 0.0001

        self.att_t1 = SPPF(self.split, self.split * 2)
        self.att_s1 = SPPF(self.split, self.split * 2)
        self.att_t2 = SPPF(self.split*2, self.split)
        self.att_s2 = SPPF(self.split * 2, self.split)

    def e(self, s):
        return torch.exp(self.clamp * (torch.sigmoid(s) * 2 - 1)) + self.affine_eps

    def forward(self, x, rev=False):
        if not rev:
            x1, x2 = x[:, 0: self.split, :, :], x[:, self.split:, :, :]

            s1, t1 = self.att_s1(x1), self.att_t1(x1)
            y2 = self.e(s1) * x2 + t1

            s2, t2 = self.att_s2(y2), self.att_t2(y2)
            y1 = self.e(s2) * x1 + t2

            out = torch.cat((y1, y2), 1)
            return out
        else:
            y1, y2 = x[:, 0: self.split, :, :], x[:, self.split:, :, :]

            s2, t2 = self.att_s2(y2), self.att_t2(y2)
            x1 = (y1 - t2) / self.e(s2)

            s1, t1 = self.att_s1(x1), self.att_t1(x1)
            x2 = (y2 - t1) / self.e(s1)

            out = torch.cat((x1, x2), 1)
            return out



class SCA1(nn.Module):
    def __init__(self, dims_in=[[48*4, 128, 128]]):
        super(SCA1, self).__init__()

        self.current_dims = dims_in

        ops1 = []
        b = SPPFCB([[36, 128, 128]], clamp=1.0, split=12)
        ops1.append(b)
        for j in range(2):
            b = RNVPCouplingBlock([[36, 128, 128]], clamp=1.0, split=12)
            ops1.append(b)

        ops2 = []
        b = SPPFCB([[36, 128, 128]], clamp=1.0, split=12)
        ops2.append(b)
        for j in range(3):
            b = RNVPCouplingBlock([[36, 128, 128]], clamp=1.0, split=12)
            ops2.append(b)

        ops3 = []
        b = SPPFCB([[36, 128, 128]], clamp=1.0, split=12)
        ops3.append(b)
        for j in range(4):
            b = RNVPCouplingBlock([[36, 128, 128]], clamp=1.0, split=12)
            ops3.append(b)

        ops4 = []
        for j in range(1):
            b = AttCouplingBlock([[36, 128, 128]], clamp=1.0, split=12)
            ops4.append(b)

        ops5 = []
        for j in range(4):
            b = RNVPCouplingBlock([[36, 128, 128]], clamp=1.0, split=12)
            ops5.append(b)

        ops6 = []
        for j in range(3):
            b = RNVPCouplingBlock([[36, 128, 128]], clamp=1.0, split=12)
            ops6.append(b)

        ops7 = []
        for j in range(2):
            b = RNVPCouplingBlock([[36, 128, 128]], clamp=1.0, split=12)
            ops7.append(b)

        self.ops1 = nn.ModuleList(ops1)
        self.ops2 = nn.ModuleList(ops2)
        self.ops3 = nn.ModuleList(ops3)
        self.ops4 = nn.ModuleList(ops4)
        self.ops5 = nn.ModuleList(ops5)
        self.ops6 = nn.ModuleList(ops6)
        self.ops7 = nn.ModuleList(ops7)

        self.down = nn.PixelUnshuffle(2)
        self.up = nn.PixelShuffle(2)

        sppf1 = []
        b = SPPFCB([[27, 128, 128]], clamp=1.0, split=9)
        sppf1.append(b)
        for j in range(2):
            b = RNVPCouplingBlock([[27, 128, 128]], clamp=1.0, split=9)
            sppf1.append(b)

        sppf2 = []
        b = SPPFCB([[27, 128, 128]], clamp=1.0, split=9)
        sppf2.append(b)
        for j in range(2):
            b = RNVPCouplingBlock([[27, 128, 128]], clamp=1.0, split=9)
            sppf2.append(b)

        self.sppf1 = nn.ModuleList(sppf1)
        self.sppf2 = nn.ModuleList(sppf2)

    def forward(self, out, rev=False):
        if not rev:
            out = self.down(out)
            for op in self.ops1:
                out = op.forward(out, False)
            u1 = out[:, 0:9, :, :]
            d1 = out[:, 9:, :, :]
            for op in self.sppf1:
                d1 = op.forward(d1, False)

            out = self.down(u1)
            for op in self.ops2:
                out = op.forward(out, False)
            u2 = out[:, 0:9, :, :]
            d2 = out[:, 9:, :, :]
            for op in self.sppf2:
                d2 = op.forward(d2, False)

            out = self.down(u2)
            for op in self.ops3:
                out = op.forward(out, False)

            for op in self.ops4:
                out = op.forward(out, False)

            for op in self.ops5:
                out = op.forward(out, False)
            out = self.up(out)

            out =torch.cat((out, d2), 1)
            for op in self.ops6:
                out = op.forward(out, False)
            out = self.up(out)

            out =torch.cat((out, d1), 1)
            for op in self.ops7:
                out = op.forward(out, False)
            out = self.up(out)
        else:
            out = self.down(out)
            for op in reversed(self.ops7):
                out = op.forward(out, True)
            u1 = out[:, 0:9, :, :]
            d1 = out[:, 9:, :, :]
            for op in reversed(self.sppf1):
                d1 = op.forward(d1, True)

            out = self.down(u1)
            for op in reversed(self.ops6):
                out = op.forward(out, True)
            u2 = out[:, 0:9, :, :]
            d2 = out[:, 9:, :, :]
            for op in reversed(self.sppf2):
                d2 = op.forward(d2, True)

            out = self.down(u2)
            for op in reversed(self.ops5):
                out = op.forward(out, True)

            for op in reversed(self.ops4):
                out = op.forward(out, True)

            for op in reversed(self.ops3):
                out = op.forward(out, True)
            out = self.up(out)

            out =torch.cat((out, d2), 1)
            for op in reversed(self.ops2):
                out = op.forward(out, True)
            out = self.up(out)

            out =torch.cat((out, d1), 1)
            for op in reversed(self.ops1):
                out = op.forward(out, True)
            out = self.up(out)

        return out


class VDM_PCD(nn.Module):
    def __init__(self):
        super(VDM_PCD, self).__init__()

        self.pcdalign = PcdAlign.PcdAlign(nf=3, groups=1)
        self.scale1 = SCA1()


    def forward(self, cur=None, ref=None, label=None, blend=1):
        out = cur
        cur1 = out
        cur2 = F.interpolate(cur1, scale_factor=0.5, mode='bilinear', align_corners=False)
        cur3 = F.interpolate(cur1, scale_factor=0.25, mode='bilinear', align_corners=False)
        cur_feats = [cur1, cur2, cur3]
        aligned = []
        # re_align = self.pcdalign(nbr_fea_l=cur_feats, ref_fea_l=cur_feats)
        # aligned.append(re_align)
        for i in range(2):
            re = ref[:, i * 3:i * 3 + 3, :, :]
            re1 = re
            re2 = F.interpolate(re1, scale_factor=0.5, mode='bilinear', align_corners=False)
            re3 = F.interpolate(re1, scale_factor=0.25, mode='bilinear', align_corners=False)
            ref_feats = [re1, re2, re3]
            re_align = self.pcdalign(nbr_fea_l=ref_feats, ref_fea_l=cur_feats)
            aligned.append(re_align)

        # for op in self.convs:
        #     cur1 = op.forward(cur1)
        out = torch.cat((cur1, aligned[0], aligned[1]), 1)

        out = self.scale1(out, False)
        res = []
        lab = []
        if label is not None:
            f = out[:, 0:3, :, :]
            z = out[:, 3:, :, :]
            rever = torch.cat((label, z), 1)
            # with torch.no_grad():
            rever = self.scale1(rever, True)

            for i in range(3):
                res.append(rever[:, i * 3:i * 3 + 3, :, :])

            lab.append(cur1)
            lab.append(aligned[0])
            lab.append(aligned[1])


            # print(res[0][0][0])
            # print("*********************************")
            # print(lab[0][0][0])


        frs = []
        for i in range(3):
            frs.append(out[:, i*3:i*3+3, :, :])



        return frs, res, lab


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """

    def __init__(self,
                 ch,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=0.1,
                 reparam_offset=2 ** -18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset

        self.build(ch)

    def build(self, ch):
        self.pedestal = self.reparam_offset ** 2
        self.beta_bound = ((self.beta_min + self.reparam_offset ** 2) ** 0.5)
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch) + self.pedestal)
        self.beta = nn.Parameter(beta)

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma)
        self.pedestal = self.pedestal

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta ** 2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma ** 2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs ** 2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs


def warp(x, flo):
    """
    From PWCNet, warp an image/tensor (im2) back to im1, according to the optical flow
    Args:
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W], pre-computed optical flow, im2-->im1
    Returns: warped image and mask indicating valid positions
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

    # if W==128:
    # np.save('mask.npy', mask.cpu().data.numpy())
    # np.save('warp.npy', output.cpu().data.numpy())

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask, mask


class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


###############################################################################################
def model_fn_decorator(loss_fn, mode='train'):

    # vgg features as region-level statistics
    l2_loss = torch.nn.MSELoss().cuda()
    l1c_loss = L1_Charbonnier_loss().cuda()
    # ssim_loss = pytorch_ssim.SSIM(window_size=11).cuda()
    # dis_model = D_Net(3).cuda()

    def val_model_fn(args, data, model, net_metric, save_path):
        model.eval()

        # prepare input and forward
        number = data['number']
        in_img = data['in_img'][0].cuda()
        label = data['label'][0].cuda()
        num_img_aux = len(data['in_img_aux'])
        assert num_img_aux > 0
        in_img_aux = data['in_img_aux'][0].cuda()
        for i in range(1, num_img_aux):
            in_img_aux = torch.cat([in_img_aux, data['in_img_aux'][i].cuda()], axis=1)

        with torch.no_grad():
            frs, res, lab = model(cur=in_img, ref=in_img_aux, label=None)

            loss = loss_fn(frs[0], label, feature_layers=[2]) + F.l1_loss(frs[0], label) + 0.25 * l1c_loss(frs[0],
                                                                                                           label)
            loss += loss_fn(frs[1], in_img - label, feature_layers=[2]) + F.l1_loss(frs[1],
                                                                                    in_img - label) + 0.25 * l1c_loss(
                frs[1], in_img - label)
            loss += loss_fn(frs[2], in_img - label, feature_layers=[2]) + F.l1_loss(frs[2],
                                                                                    in_img - label) + 0.25 * l1c_loss(
                frs[2], in_img - label)
            # for i in range(3):
            #     loss += loss_fn(res[i], lab[i])

            # loss += 1 - ssim(out_img, label)
            out_img = frs[0]
            # print(out_img)

        out_put = tensor2img(out_img)
        gt = tensor2img(label)
        pre = torch.clamp(out_img, min=0, max=1)
        tar = torch.clamp(label, min=0, max=1)

        # Calculate LPIPS
        cur_lpips = net_metric.forward(pre, tar, normalize=True)
        # Calculate PSNR
        cur_psnr = calculate_psnr(out_put, gt)
        # Calculate SSIM
        cur_ssim = calculate_ssim(out_put, gt)

        # save images
        if args.SAVE_IMG != 0:
            out_save = out_img.detach().cpu()
            torchvision.utils.save_image(out_save, save_path + '/' + 'val_%05s' % number[0] + '.%s' % args.SAVE_IMG)

        return loss, cur_psnr, cur_ssim, cur_lpips.item()


    def test_model_fn(args, data, model, save_path):
        model.eval()

        # prepare input and forward
        number = data['number']
        in_img = data['in_img'][0].cuda()
        num_img_aux = len(data['in_img_aux'])
        assert num_img_aux > 0
        in_img_aux = data['in_img_aux'][0].cuda()
        for i in range(1, num_img_aux):
            in_img_aux = torch.cat([in_img_aux, data['in_img_aux'][i].cuda()], axis=1)

        with torch.no_grad():
            frs, res, lab = model(cur=in_img, ref=in_img_aux)
            out_img = frs

        # save images
        if args.SAVE_IMG != 0:
            for i in range(3):
                out_save = out_img[i].detach().cpu()
                torchvision.utils.save_image(out_save, save_path + '/' + 'test_%05s' % number[0] + '.%s' % args.SAVE_IMG)


    def train_model_fn(args, data, model, iters, epoch):
        model.train()
        in_img = data['in_img'][0].cuda()
        label = data['label'][0].cuda()
        in_img_aux = data['in_img_aux'][0].cuda()
        for i in range(1, args.NUM_AUX_FRAMES):
            in_img_aux = torch.cat([in_img_aux, data['in_img_aux'][i].cuda()], axis=1)

        # if epoch >1:
        #     lab = label
        # else:
        #     lab = Noneone

        frs, res, lab = model(cur=in_img, ref=in_img_aux, label=None)

        loss = loss_fn(frs[0], label, feature_layers=[2]) + F.l1_loss(frs[0], label) + 0.25 * l1c_loss(frs[0], label)
        loss += loss_fn(frs[1], in_img - label, feature_layers=[2]) + F.l1_loss(frs[1], in_img - label) + 0.25 * l1c_loss(frs[1], in_img - label)
        loss += loss_fn(frs[2], in_img - label, feature_layers=[2]) + F.l1_loss(frs[2], in_img - label) + 0.25 * l1c_loss(frs[2], in_img - label)
        # reverse loss
        # for i in range(3):
        #     loss += 0.1 * l1c_loss(res[i], lab[i])
        out_img = frs[0]

        out_put = tensor2img(out_img)
        gt = tensor2img(label)
        # Calculate PSNR
        cur_psnr = calculate_psnr(out_put, gt)

        loss_temporal = loss*0
        loss_reg = loss*0
        # loss += dis_loss

        # save images
        if iters % args.SAVE_ITER == (args.SAVE_ITER - 1):
            in_save = in_img.detach().cpu()[:, 0:3, :, :]
            out_save = out_img.detach().cpu()
            gt_save = label.detach().cpu()
            res_save = torch.cat((in_save, out_save, gt_save), 2)
            save_number = (iters + 1) // args.SAVE_ITER
            torchvision.utils.save_image(res_save, args.VISUALS_DIR + '/visual_x%04d_' % args.SAVE_ITER + '%05d' % save_number + '.jpg')

        return loss, loss_temporal, loss_reg, cur_psnr

    if mode == 'test':
        fn = test_model_fn
    elif mode == 'val':
        fn = val_model_fn
    else:
        fn = train_model_fn
    return fn













