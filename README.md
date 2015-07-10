# stnbhwd
Modules for spatial transformer networks (BHWD layout)

```
require 'stn'

nn.AffineGridGeneratorBHWD(height, width)
-- takes B x 2 x 3 affine transform matrices as input, 
-- outputs a height x width grid in normalized [-1,1] coordinates
-- output layout is B,H,W,2 where the first coordinate in the 4th dimension is y, and the second is x

nn.BilinearSamplerBHWD()
-- takes a table {inputImages, grids} as inputs
-- outputs the interpolated images according to the grids
-- inputImages is a batch of samples in BHWD layout
-- grids is a batch of grids (output of the other module)
-- output is also BHWD
```

These modules should let one implement the Spatial Transformer Networks (Jaderberg et al.)
http://arxiv.org/abs/1506.02025
