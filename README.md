# stn3d

## Installation

``` bash
luarocks make stn3d-scm-1.rockspec
```

## Modules

These are the basic modules (BTHWC layout) needed to implement a 3D variant of Spatial Transformer Network (Jaderberg et al.) http://arxiv.org/abs/1506.02025

``` lua
require 'stn3d'

nn.Affine3dGridGeneratorBTHWC(depth, height, width)
-- takes B x 3 x 4 affine transform matrices as input,
-- outputs a height x width grid in normalized [-1,1] coordinates
-- output layout is B,T,H,W,3 where the first coordinate in the 5th dimension is z, and the second is y, third in x

nn.TrilinearSamplerBTHWC()
-- takes a table {inputVolumes, grids} as inputs
-- outputs the interpolated volumes according to the grids
-- inputImages is a batch of samples in BTHWC layout
-- grids is a batch of grids (output of Affine3dGridGeneratorBTWC)
-- output is also BTHWC
```

## Advanced module

This module allows the user to put a constraint on the possible transformations.
It should be placed between the localisation network and the grid generator.

``` lua
require 'stn3d'

nn.Affine3dTransformMatrixGenerator(useScale, useTranslation)
-- takes a B x nbParams tensor as inputs
-- nbParams depends on the contrained transformation
-- The parameters for the selected transformation(s) should be supplied in the
-- following order: scaleFactor, translationZ, translationY, translationX
-- If no transformation is specified, it generates a generic affine transformation (nbParams = 12)
-- outputs B x 3 x 4 affine transform matrices
```


If this code is useful to your research, please cite this repository.

## Acknowledgements
This code is derived from the excellent [2D Spatial Transformer implementation](https://github.com/qassemoquab/stnbhwd) by [@qassemoquab](https://github.com/qassemoquab).
