require 'nn'
require 'cutorch'
require 'libstn3d'
require 'libcustn3d'

include('Affine3dTransformMatrixGenerator.lua')
--include('AffineGridGeneratorBHWD.lua')
include('Affine3dGridGeneratorBTHWC.lua')
--include('BilinearSamplerBHWD.lua')
include('TrilinearSamplerBTHWC.lua')

--include('test.lua')

return nn
