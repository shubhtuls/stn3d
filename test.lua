-- you can easily test specific units like this:
-- th -lnn -e "nn.test{'LookupTable'}"
-- th -lnn -e "nn.test{'LookupTable', 'Add'}"

local mytester = torch.Tester()
local jac
local sjac

local precision = 1e-5
local expprecision = 1e-4

local stn3dtest = {}

function stn3dtest.TrilinearSamplerBTHWC_batch()
   local nframes = torch.random(2,10)
   local nTime = torch.random(1,5)
   local height = torch.random(1,5)
   local width = torch.random(1,5)
   local channels = torch.random(1,6)
   local inputImages = torch.zeros(nframes, nTime, height, width, channels):uniform()
   local grids = torch.zeros(nframes, nTime, height, width, 3):uniform()
   local module = nn.TrilinearSamplerBTHWC()

   -- test input images (first element of input table)
   module._updateOutput = module.updateOutput
   function module:updateOutput(input)
      return self:_updateOutput({input, grids})
   end
   
   module._updateGradInput = module.updateGradInput
   function module:updateGradInput(input, gradOutput)
      self:_updateGradInput({input, grids}, gradOutput)
      return self.gradInput[1]
   end

   local errImages = jac.testJacobian(module,inputImages)
   mytester:assertlt(errImages,precision, 'error on state ')

   -- test grids (second element of input table)
   function module:updateOutput(input)
      return self:_updateOutput({inputImages, input})
   end
   
   function module:updateGradInput(input, gradOutput)
      self:_updateGradInput({inputImages, input}, gradOutput)
      return self.gradInput[2]
   end

   local errGrids = jac.testJacobian(module,grids)
   print(errImages,errGrids)
   mytester:assertlt(errGrids,precision, 'error on state ')
end

function stn3dtest.TrilinearSamplerBTHWC_single()
   local nTime = torch.random(1,5)
   local height = torch.random(1,5)
   local width = torch.random(1,5)
   local channels = torch.random(1,6)
   local inputImages = torch.zeros(nTime, height, width, channels):uniform()
   local grids = torch.zeros(nTime, height, width, 3):uniform()
   local module = nn.TrilinearSamplerBTHWC()

   -- test input images (first element of input table)
   module._updateOutput = module.updateOutput
   function module:updateOutput(input)
      return self:_updateOutput({input, grids})
   end
   
   module._updateGradInput = module.updateGradInput
   function module:updateGradInput(input, gradOutput)
      self:_updateGradInput({input, grids}, gradOutput)
      return self.gradInput[1]
   end

   local errImages = jac.testJacobian(module,inputImages)
   mytester:assertlt(errImages,precision, 'error on state ')

   -- test grids (second element of input table)
   function module:updateOutput(input)
      return self:_updateOutput({inputImages, input})
   end
   
   function module:updateGradInput(input, gradOutput)
      self:_updateGradInput({inputImages, input}, gradOutput)
      return self.gradInput[2]
   end

   local errGrids = jac.testJacobian(module,grids)
   print(errImages,errGrids)
   mytester:assertlt(errGrids,precision, 'error on state ')
end

mytester:add(stn3dtest)

if not nn then
   require 'nn'
   jac = nn.Jacobian
   sjac = nn.SparseJacobian
   mytester:run()
else
   jac = nn.Jacobian
   sjac = nn.SparseJacobian
   function stn3d.test(tests)
      -- randomize stuff
      math.randomseed(os.time())
      mytester:run(tests)
      return mytester
   end
end