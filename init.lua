require 'nn'
require 'cutorch'
require 'libstn3d'
require 'libcustn3d'

include('Affine3dTransformMatrixGenerator.lua')
include('Affine3dGridGeneratorBTHWC.lua')
include('TrilinearSamplerBTHWC.lua')

return nn