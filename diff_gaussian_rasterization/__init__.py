#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    seg_encoding,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        seg_encoding,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

def rasterize_histograms(
    geom_buffer,
    binning_buffer,
    img_buffer,
    num_points,
    num_rendered,
    image_height,
    image_width,
    error_bins,
    num_bins,
    mass_buffer,
    error_hist,
):
    """Accumulate per-Gaussian evidence into mass_buffer [P] and error_hist [P, B].

    Replays the sorted geometry from the forward pass that produced the buffers
    (no re-render). mass_buffer and error_hist are mutated in place, so both must
    be contiguous fp32 CUDA tensors. error_bins is an int32 [H, W] map of per-pixel
    error bin ids. Runs under no autograd; call inside torch.no_grad().
    """
    return _C.rasterize_histograms(
        geom_buffer,
        binning_buffer,
        img_buffer,
        int(num_points),
        int(num_rendered),
        int(image_height),
        int(image_width),
        error_bins,
        int(num_bins),
        mass_buffer,
        error_hist,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        seg_encoding,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            raster_settings.gt_segmentation,
            raster_settings.num_segmentation_classes,
            raster_settings.vote_buffer,
            means3D,
            colors_precomp,
            seg_encoding,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        num_rendered, color, rendered_seg, radii, geomBuffer, binningBuffer, imgBuffer, invdepths = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, seg_encoding, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer)
        # geom/binning/image buffers are exposed as outputs so callers can run a
        # follow-up histogram pass (RasterizeHistogramsCUDA) that reuses the sorted
        # geometry without re-rendering. num_rendered is returned as a 1-element
        # tensor (non-differentiable metadata).
        num_rendered_t = torch.tensor(num_rendered, dtype=torch.int32)
        return color, rendered_seg, radii, invdepths, geomBuffer, binningBuffer, imgBuffer, num_rendered_t

    @staticmethod
    def backward(ctx, *grad_outputs):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, seg_encoding, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Forward outputs: (color, rendered_seg, radii, invdepths, geom, binning,
        # img, num_rendered); only color/seg/invdepths are differentiable.
        grad_out_color, grad_out_seg, grad_out_depth = grad_outputs[0], grad_outputs[1], grad_outputs[3]

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D,
                radii,
                colors_precomp,
                seg_encoding,
                opacities,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                grad_out_color,
                grad_out_depth,
                grad_out_seg,
                sh,
                raster_settings.sh_degree,
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.antialiasing,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_seg_encoding = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,          # means3D
            grad_means2D,          # means2D
            grad_sh,               # sh
            grad_colors_precomp,   # colors_precomp
            grad_seg_encoding,     # seg_encoding (must match forward's input order)
            grad_opacities,        # opacities
            grad_scales,           # scales
            grad_rotations,        # rotations
            grad_cov3Ds_precomp,   # cov3Ds_precomp
            None,                  # raster_settings
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    gt_segmentation : torch.Tensor
    num_segmentation_classes : int
    vote_buffer : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    antialiasing : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)

        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None, seg_encoding = None):

        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if seg_encoding is None:
            raise Exception('seg_encoding is required: pass pc.get_segmentation_encoding')

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            seg_encoding,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )
