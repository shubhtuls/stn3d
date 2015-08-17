require 'nn'
require 'cutorch'
require 'libstn'
require 'libcustn'

include('AffineTransformMatrixGenerator.lua')
include('AffineGridGeneratorBHWD.lua')
include('BilinearSamplerBHWD.lua')

include('test.lua')

return nn
