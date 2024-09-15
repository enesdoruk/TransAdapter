import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.model_parts import window_partition, window_reverse, PatchMerging, PatchEmbed
from utils.lossZoo import adv_local
import random
from torch.nn import KLDivLoss
import numpy as np
import torch.nn.functional as F

try:
    import os, sys

    kernel_path = os.path.abspath(os.path.join('..'))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse
except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 3)
    B, P, D = size[:3]
    feat_var = feat.view(B, P, -1).var(dim=1) + eps
    feat_std = feat_var.sqrt().view(B, 1, D)
    feat_mean = feat.view(B, P, -1).mean(dim=1).view(B, 1, D)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


class CrossFeatTrans(nn.Module):
    def __init__(self, in_planes):
        super(CrossFeatTrans, self).__init__()

        self.in_planes = in_planes

        self.f = nn.Linear(self.in_planes, self.in_planes, (1, 1))
        self.g = nn.Linear(self.in_planes, self.in_planes, (1, 1))
        self.h = nn.Linear(self.in_planes, self.in_planes, (1, 1))
        self.sm = nn.Softmax(dim = -1)
        self.prj = nn.Linear(self.in_planes, self.in_planes, (1, 1))

        self.gating_param = nn.Parameter(torch.ones(self.in_planes), requires_grad= True).cuda()

    def forward(self, source, target):
        b, p, d = source.size()

        src = self.f(mean_variance_norm(source))
        trgt = self.g(mean_variance_norm(target))

        s2t = self.sm(torch.bmm(src.permute(0, 2, 1), trgt))
        t2s = self.sm(torch.bmm(trgt.permute(0, 2, 1), src))

        gating = (self.gating_param).view(1,1,-1,1)
        attn = (1.-torch.sigmoid(gating)) * s2t + torch.sigmoid(gating) * t2s

        source_v = self.h(source)
        feattrans = torch.bmm(source_v, attn.squeeze(0).permute(0, 2, 1))
        
        feattrans = feattrans.view(b, p, d)
        feattrans = self.prj(feattrans)
        
        pair_dist = torch.nn.functional.pairwise_distance(source, target)
        
        out = (feattrans * pair_dist.unsqueeze(-1))

        return out
    

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.fc1(x)
        x1 = self.act(x1)
        x1 = self.drop(x1)

        x2 = self.fc2(x1)
        x2 = self.drop(x2)

        return x2


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.fc1 = nn.Linear(self.dim*2, self.dim*2)
        self.fc_rel = nn.ReLU()
        
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2, x_t1=None, x_t2=None, mask1=None, mask2=None, ad_net=None, is_train=False):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x1.shape

        if is_train:
            xcat = torch.cat((x_t1, x_t2), dim=2)
            xcat = self.fc1(xcat)
            xcat = self.fc1(xcat)
            x_t_lin1, x_t_lin2 =  xcat[:,:,self.dim:], xcat[:,:,:self.dim]

            x_t1 = x_t1 + x_t_lin1
            x_t1 = self.fc_rel(x_t1)
            x_t2 = x_t2 + x_t_lin2
            x_t2 = self.fc_rel(x_t2)
             
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        qkv1 = self.qkv(x1).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2] 
        q1 = q1 * self.scale

        attn1 = (q1 @ k1.transpose(-2, -1))
        attn1 = attn1 + relative_position_bias.unsqueeze(0)

        qkv2 = self.qkv(x2).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2] 
        q2 = q2 * self.scale

        attn2 = (q1 @ k2.transpose(-2, -1))
        attn2 = attn2 + relative_position_bias.unsqueeze(0)

        if is_train:
            qkv_t1 = self.qkv(x_t1).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q_t1, k_t1, v_t1 = qkv_t1[0], qkv_t1[1], qkv_t1[2] 
            q_t1 = q_t1 * self.scale

            attn_t1 = (q_t1 @ k_t1.transpose(-2, -1))
            attn_t1 = attn_t1 + relative_position_bias.unsqueeze(0)

            qkv_t2 = self.qkv(x_t2).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q_t2, k_t2, v_t2 = qkv_t2[0], qkv_t2[1], qkv_t2[2] 
            q_t2 = q_t2 * self.scale

            attn_t2 = (q_t1 @ k_t2.transpose(-2, -1))
            attn_t2 = attn_t2 + relative_position_bias.unsqueeze(0)

        if mask1 is not None: 
            nW = mask1.shape[0]
            attn1 = attn1.view(B_ // nW, nW, self.num_heads, N, N) + mask1.unsqueeze(1).unsqueeze(0)
            attn1 = attn1.view(-1, self.num_heads, N, N)

            attn2 = attn2.view(B_ // nW, nW, self.num_heads, N, N) + mask2.unsqueeze(1).unsqueeze(0)
            attn2 = attn2.view(-1, self.num_heads, N, N)

            attn = self.softmax(torch.cat((attn1, attn2), dim=1))

            if is_train:
                attn_t1 = attn_t1.view(B_ // nW, nW, self.num_heads, N, N) + mask1.unsqueeze(1).unsqueeze(0)
                attn_t1 = attn_t1.view(-1, self.num_heads, N, N)

                attn_t2 = attn_t2.view(B_ // nW, nW, self.num_heads, N, N) + mask2.unsqueeze(1).unsqueeze(0)
                attn_t2 = attn_t2.view(-1, self.num_heads, N, N)
 
                attn_t = self.softmax(torch.cat((attn_t1, attn_t2), dim=1))
        else:
            attn = self.softmax(torch.cat((attn1, attn2), dim=1))
            if is_train:
                attn_t = self.softmax(torch.cat((attn_t1, attn_t2), dim=1))

        loss_ad, loss_ad_t = 0, 0
        if ad_net is not None:
            eps=1e-10
            batch_size = k1.size(0)
            ad_out, loss_ad, ad_feat = adv_local(k1, k2, ad_net=ad_net, is_source=True)
            entropy = - ad_feat * torch.log2(ad_feat + eps) - (1.0 - ad_feat) * torch.log2(1.0 - ad_feat + eps)
            entropy = entropy.contiguous().view(batch_size, entropy.shape[2], 1, entropy.shape[1])
            attn = torch.cat((attn[:,:,0,:].unsqueeze(2) * entropy, attn[:,:,1:,:]), 2)
            if is_train:
                ad_out_t, loss_ad_t, ad_feat_t = adv_local(k_t1, k_t2, ad_net=ad_net, is_source=False)
                entropy_t = - ad_feat_t * torch.log2(ad_feat_t + eps) - (1.0 - ad_feat_t) * torch.log2(1.0 - ad_feat_t + eps)
                entropy_t = entropy_t.contiguous().view(batch_size, entropy_t.shape[2], 1, entropy_t.shape[1])
                attn_t = torch.cat((attn_t[:,:,0,:].unsqueeze(2) * entropy_t, attn_t[:,:,1:,:]), 2)

        attn = self.attn_drop(attn)
        x = (attn @ torch.cat((v1, v2), dim=1)).transpose(1, 2).reshape(B_, N, C*2)
        weights = x
        x = x[:, :, :C]
        x = self.proj(x)
        x = self.proj_drop(x)

        if is_train:
            attn_t = self.attn_drop(attn_t)
            x_t = (attn_t @ torch.cat((v_t1, v_t2), dim=1)).transpose(1, 2).reshape(B_, N, C*2)
            weights_t = x_t
            x_t = x_t[:, :, :C]
            x_t = self.proj(x_t)
            x_t = self.proj_drop(x_t)

        if is_train:
            return x, x_t, weights, weights_t, loss_ad, loss_ad_t
        else:
            return x, None, weights, None, loss_ad, None

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            attn_mask2 = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask2 = attn_mask2.masked_fill(attn_mask2 != 0, float(-100.0)).masked_fill(attn_mask2 == 0, float(0.0))
        else:
            attn_mask = None
            attn_mask2 = None

        self.register_buffer("attn_mask", attn_mask)
        self.register_buffer("attn_mask2", attn_mask2)
        self.fused_window_process = fused_window_process

    def forward(self, x, x_t=None, ad_net=None, is_train=False):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if is_train:
            shortcut_t = x_t
            x_t = self.norm1(x_t)
            x_t = x_t.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                x1 = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
                x2 = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                x2 = window_partition(x2, self.window_size)  # nW*B, window_size, window_size, C
                if is_train:
                    x_t1 = window_partition(x_t, self.window_size)  # nW*B, window_size, window_size, C
                    x_t2 = torch.roll(x_t, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                    x_t2 = window_partition(x_t2, self.window_size)  # nW*B, window_size, window_size, C
            else:
                x1 = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
                x2 = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
                if is_train:
                    x_t1 = window_partition(x_t, self.window_size)  # nW*B, window_size, window_size, C
                    x_t2 = WindowProcess.apply(x_t, B, H, W, C, -self.shift_size, self.window_size)
        else:
            x1 = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
            x2 = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
            if is_train:
                x_t1 = window_partition(x_t, self.window_size)  # nW*B, window_size, window_size, C
                x_t2 = window_partition(x_t, self.window_size)  # nW*B, window_size, window_size, C

        x1_windows = x1.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        x2_windows = x2.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        if is_train:
            x_windows_t1 = x_t1.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
            x_windows_t2 = x_t2.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
            
            # W-MSA/SW-MSA
            attn_windows, attn_windows_t, att_weights, att_weights_t,\
                  loss_ad, loss_ad_t = self.attn(x1=x2_windows, x2=x1_windows, x_t1=x_windows_t2, x_t2=x_windows_t1, \
                                                 mask1=self.attn_mask, mask2=self.attn_mask2, ad_net=ad_net, is_train=is_train)  # nW*B, window_size*window_size, C
            # merge windows
            attn_windows_t = attn_windows_t.view(-1, self.window_size, self.window_size, C)
        else:
            # W-MSA/SW-MSA
            attn_windows, _, att_weights, _, loss_ad, _ = self.attn(x1=x2_windows, x2=x1_windows, mask1=self.attn_mask, \
                                                                    mask2=self.attn_mask2, ad_net=ad_net, is_train=is_train)  # nW*B, window_size*window_size, C
        
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        
        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                att_weights = window_reverse(att_weights, self.window_size, H, W)
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                if is_train:
                    shifted_x_t = window_reverse(attn_windows_t, self.window_size, H, W)  # B H' W' C
                    att_weights_t = window_reverse(att_weights_t, self.window_size, H, W)
                    x_t = torch.roll(shifted_x_t, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
                att_weights = window_reverse(att_weights, self.window_size, H, W)
                if is_train:
                    att_weights_t = window_reverse(att_weights_t, self.window_size, H, W)
                    x_t = WindowProcessReverse.apply(attn_windows_t, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            att_weights = window_reverse(att_weights, self.window_size, H, W)
            x = shifted_x
            if is_train:
                shifted_x_t = window_reverse(attn_windows_t, self.window_size, H, W)  # B H' W' C
                att_weights_t = window_reverse(att_weights_t, self.window_size, H, W)
                x_t = shifted_x_t

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if is_train:
            x_t = x_t.view(B, H * W, C)
            x_t = shortcut_t + self.drop_path(x_t)
            # FFN
            x_t = x_t + self.drop_path(self.mlp(self.norm2(x_t)))

            return x, x_t, att_weights, att_weights_t, loss_ad, loss_ad_t  
        else: 
            return x, None, att_weights, None, loss_ad, None

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops
    

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False, shift_size=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.shift_size = shift_size

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process)
            for i in range(depth)])
        
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_t=None, ad_net=None, is_train=False):
        if is_train:
            for blk in self.blocks:
                if self.use_checkpoint:
                    x, attn_mat = checkpoint.checkpoint(blk, x)
                    x_t, attn_mat_t = checkpoint.checkpoint(blk, x_t)
                else:
                    x, x_t, attn_mat, attn_mat_t, loss_ad, loss_ad_t = blk(x, x_t, ad_net, is_train)

            if self.downsample is not None:
                x_t = self.downsample(x_t)
        else:
            for blk in self.blocks:
                if self.use_checkpoint:
                    x, attn_mat = checkpoint.checkpoint(blk, x)
                else:
                    x, _, attn_mat, _, loss_ad, _ = blk(x=x, ad_net=ad_net, is_train=is_train)
                
        if self.downsample is not None:
            x = self.downsample(x)

        if is_train:
            return x, x_t, attn_mat, attn_mat_t, loss_ad, loss_ad_t
        else:
            return x, None, attn_mat, None, loss_ad, None
            
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class Swin(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, shift_size=0, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.shift_size = shift_size
        self.num_heads = num_heads

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.param_trans = nn.ModuleList()
        for i_layer in range(self.num_layers):
            paramtrans = CrossFeatTrans(in_planes=int(embed_dim * 2 ** i_layer),
                                        )
            self.param_trans.append(paramtrans)
        paramtrans = CrossFeatTrans(in_planes=int(embed_dim * 2 ** i_layer),
                                     )
        self.param_trans.append(paramtrans)

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               fused_window_process=fused_window_process,
                               shift_size=self.shift_size)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.kl_div = KLDivLoss(reduction="batchmean")

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x, x_t= None, ad_net=None, is_train=False):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
    
        if is_train:
            x_t = self.patch_embed(x_t)
            if self.ape:
                x_t = x_t + self.absolute_pos_embed
            x_t = self.pos_drop(x_t)    

            x_t = self.param_trans[0](x_t, x)
            
            param_rand = random.choice(np.arange(self.num_layers).tolist())

            attn_mats = []
            attn_mats_t = []
            for i, layer in enumerate(self.layers):
                B = x_t.shape[0]

                if i == len(self.layers) - 1: 
                    x, x_t, attn_mat, attn_mat_t, ad_loss, ad_loss_t = layer(x, x_t, ad_net, is_train=is_train)
                else:
                    x, x_t, attn_mat, attn_mat_t, ad_loss, ad_loss_t = layer(x, x_t, is_train=is_train)

                attn_mats.append(attn_mat)
                attn_mats_t.append(attn_mat_t)

                if i == param_rand:
                    x_t = self.param_trans[param_rand+1](x_t, x)

            x_t = self.norm(x_t)  # B L C
            x_t_glb = x_t
            x_t = self.avgpool(x_t.transpose(1, 2))  # B C 1
            x_t = torch.flatten(x_t, 1)

        else:
            attn_mats = []
            for i, layer in enumerate(self.layers):
                if i == len(self.layers) - 1: 
                    x, _, attn_mat, _, ad_loss, _ = layer(x, ad_net=ad_net, is_train=is_train)
                else:
                    x, _, attn_mat, _, ad_loss, _  = layer(x, is_train=is_train)
                attn_mats.append(attn_mat)

        x = self.norm(x)  # B L C
        x_glb = x
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)

        if is_train:
            kl_div = self.kl_div(torch.log(torch.softmax(x, dim=1)), torch.softmax(x_t, dim=1))
            return x, x_glb, x_t, x_t_glb, attn_mats, attn_mats_t, ad_loss, ad_loss_t, kl_div
        else:
            return x, x_glb,  None, None, attn_mats, None, ad_loss, None, None

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


class TransAdapter(nn.Module):
    def __init__(self, config):
        super(TransAdapter, self).__init__()

        self.img_size=config.IMG_SIZE
        self.patch_size=config.PATCH_SIZE
        self.in_chans=config.IN_CHANS
        self.num_classes=config.NUM_CLASSES
        self.embed_dim=config.EMBED_DIM
        self.depths=config.DEPTHS
        self.num_heads=config.NUM_HEADS
        self.window_size=config.WINDOW_SIZE
        self.mlp_ratio=config.MLP_RATIO
        self.qkv_bias=config.QKV_BIAS
        self.qk_scale=config.QK_SCALE
        self.drop_rate=config.DROP_RATE
        self.attn_drop_rate=config.ATTN_DROP_RATE
        self.drop_path_rate=config.DROP_PATH_RATE
        self.norm_layer=config.NORM_LAYER
        self.ape=config.APE
        self.patch_norm=config.PATCH_NORM
        self.use_checkpoint=config.USE_CHECKPOINT
        self.fused_window_process=config.FUSED_WINDOW_PROCESS
        self.shift_size = config.SHIFT_SIZE
        
        self.num_layers = len(self.depths)
        self.num_features = int(self.embed_dim * 2 ** (self.num_layers - 1))

        self.transformer = Swin(img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, num_classes=self.num_classes,
                            embed_dim=self.embed_dim, depths=self.depths, num_heads=self.num_heads,
                            window_size=self.window_size, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                            drop_rate=self.drop_rate, attn_drop_rate=self.attn_drop_rate, drop_path_rate=self.drop_path_rate,
                            norm_layer=self.norm_layer, ape=self.ape, patch_norm=self.patch_norm,
                            use_checkpoint=self.use_checkpoint, fused_window_process=self.fused_window_process,
                            shift_size=self.shift_size)
        
        self.head = nn.Linear(self.num_features, self.num_classes) if self.num_classes > 0 else nn.Identity()

    def forward(self, x, x_t=None, ad_net=None, is_train=False):
        if is_train:       
            feats, feats_glb, feats_t, feats_t_glb, attn_mats,  attn_mats_t ,loss_ad, \
                loss_ad_t, loss_kl = self.transformer(x, x_t, ad_net, is_train=is_train)
            
            out = self.head(feats)

            return out, feats, feats_glb, feats_t, feats_t_glb, attn_mats, attn_mats_t, (loss_ad + loss_ad_t) / 2.0, loss_kl
        else:
            feats, feats_glb, _, _, attn_mats,  _ ,loss_ad, _, _ = self.transformer(x=x, ad_net=ad_net, is_train=is_train)
            out = self.head(feats)
            return out, feats, feats_glb, None, None, attn_mats,  None ,loss_ad, None