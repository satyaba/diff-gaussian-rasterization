/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
#define NUM_SEG_CHANNELS 32 // Per-Gaussian segmentation encoding dimensionality (locked; changing requires a rebuild)
// Compute the in-kernel photometric-only 2D-mean gradient (dL_dmean2D_photo) that
// drives SGDense's decoupled densification trigger. Set to 0 to build the backward
// WITHOUT it (diagnostic only: the trigger then reads zeros, so densification is off).
#define SEG_PHOTO_GRAD 1
#define BLOCK_X 16
#define BLOCK_Y 16

#endif