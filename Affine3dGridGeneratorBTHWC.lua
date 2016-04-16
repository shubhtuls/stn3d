local AGG, parent = torch.class('nn.Affine3dGridGeneratorBTHWC', 'nn.Module')

--[[
   Affine3dGridGeneratorBTHWC(depth, height, width) :
   Affine3dGridGeneratorBTHWC:updateOutput(transformMatrix)
   Affine3dGridGeneratorBTHWC:updateGradInput(transformMatrix, gradGrids)

   Affine3dGridGeneratorBTHWC will take 3x4 an affine image transform matrix (homogeneous 
   coordinates) as input, and output a grid, in normalized coordinates* that, once used
   with the Bilinear Sampler, will result in an affine transform.

   AffineGridGenerator 
   - takes (B,3,4)-shaped transform matrices as input (B=batch).
   - outputs a grid in BTHWC layout, that can be used directly with TrilinearSamplerBTHWC
   - initialization of the previous layer should biased towards the identity transform :
      | 1  0  0  0|
      | 0  1  0  0|
      | 0  0  1  0|
   *: normalized coordinates [-1,1] correspond to the boundaries of the input image. 
]]

function AGG:__init(depth, height, width)
   parent.__init(self)
   assert(depth > 1)
   assert(height > 1)
   assert(width > 1)
   self.depth = depth
   self.height = height
   self.width = width
   
   self.baseGrid = torch.Tensor(depth, height, width, 4)
   for i=1,self.depth do
      self.baseGrid:select(4,1):select(1,i):fill(-1 + (i-1)/(self.depth-1) * 2)
   end
   for i=1,self.height do
      self.baseGrid:select(4,2):select(2,i):fill(-1 + (i-1)/(self.height-1) * 2)
   end
   for j=1,self.width do
      self.baseGrid:select(4,3):select(3,j):fill(-1 + (j-1)/(self.width-1) * 2)
   end
   self.baseGrid:select(4,4):fill(1)
   self.batchGrid = torch.Tensor(1, depth, height, width, 4):copy(self.baseGrid)
end

local function addOuterDim(t)
   local sizes = t:size()
   local newsizes = torch.LongStorage(sizes:size()+1)
   newsizes[1]=1
   for i=1,sizes:size() do
      newsizes[i+1]=sizes[i]
   end
   return t:view(newsizes)
end

function AGG:updateOutput(_transformMatrix)
   local transformMatrix
   if _transformMatrix:nDimension()==2 then
      transformMatrix = addOuterDim(_transformMatrix)
   else
      transformMatrix = _transformMatrix
   end
   assert(transformMatrix:nDimension()==3
          and transformMatrix:size(2)==3
          and transformMatrix:size(3)==4
          , 'please input affine transform matrices (bx2x3)')
   local batchsize = transformMatrix:size(1)
   
   if self.batchGrid:size(1) ~= batchsize then
      self.batchGrid:resize(batchsize, self.depth, self.height, self.width, 4)
      for i=1,batchsize do
         self.batchGrid:select(1,i):copy(self.baseGrid)
      end
   end

   self.output:resize(batchsize, self.depth, self.height, self.width, 3)
   local flattenedBatchGrid = self.batchGrid:view(batchsize, self.depth*self.width*self.height, 4)
   local flattenedOutput = self.output:view(batchsize, self.depth*self.width*self.height, 3)
   torch.bmm(flattenedOutput, flattenedBatchGrid, transformMatrix:transpose(2,3))
   if _transformMatrix:nDimension()==2 then
      self.output = self.output:select(1,1)
   end
   return self.output
end

function AGG:updateGradInput(_transformMatrix, _gradGrid)
   local transformMatrix, gradGrid
   if _transformMatrix:nDimension()==2 then
      transformMatrix = addOuterDim(_transformMatrix)
      gradGrid = addOuterDim(_gradGrid)
   else
      transformMatrix = _transformMatrix
      gradGrid = _gradGrid
   end

   local batchsize = transformMatrix:size(1)
   local flattenedGradGrid = gradGrid:view(batchsize, self.depth*self.width*self.height, 3)
   local flattenedBatchGrid = self.batchGrid:view(batchsize, self.depth*self.width*self.height, 4)
   self.gradInput:resizeAs(transformMatrix):zero()
   self.gradInput:baddbmm(flattenedGradGrid:transpose(2,3), flattenedBatchGrid)
   -- torch.baddbmm doesn't work on cudatensors for some reason

   if _transformMatrix:nDimension()==2 then
      self.gradInput = self.gradInput:select(1,1)
   end

   return self.gradInput
end
