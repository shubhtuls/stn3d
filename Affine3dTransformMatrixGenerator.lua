local ATMG, parent = torch.class('nn.Affine3dTransformMatrixGenerator', 'nn.Module')

--[[
Affine3dTransformMatrixGenerator(useScale, useTranslation) :
AffineTransformMatrixGenerator:updateOutput(transformParams)
AffineTransformMatrixGenerator:updateGradInput(transformParams, gradParams)

This module can be used in between the localisation network (that outputs the
parameters of the transformation) and the Affine3dGridGeneratorBTHWC (that expects
an affine transform matrix as input).

The goal is to be able to use only specific transformations or a combination of them.

If no specific transformation is specified, it uses a fully parametrized
linear transformation and thus expects 12 parameters as input. In this case
the module is equivalent to nn.View(3,4):setNumInputDims(2).

Any combination of the 2 transformations (scale and/or translation)
can be used. The transform parameters must be supplied in the following order:
rotation (3 params), scale (1 param) then translation (3 params).

Important Note : The order in which the transformations are applied is different from the original STN code.
Here, we will first rotate, then scale, and finally translate.

]]

function ATMG:__init(useScale, useTranslation)
  parent.__init(self)

  -- if no specific transformation, use fully parametrized version
  self.fullMode = not(useScale or useTranslation)

  if not self.fullMode then
    self.useScale = useScale
    self.useTranslation = useTranslation
    
  end
    
end

function ATMG:check(input)
  if self.fullMode then
    assert(input:size(2)==12, 'Expected 12 parameters, got ' .. input:size(2))
  else
    local numberParameters = 0
    if self.useScale then
      numberParameters = numberParameters + 1
    end
    if self.useTranslation then
      numberParameters = numberParameters + 3
    end
    assert(input:size(2)==numberParameters, 'Expected '..numberParameters..
                                            ' parameters, got ' .. input:size(2))
  end
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

function ATMG:updateOutput(_tranformParams)
  local transformParams
  if _tranformParams:nDimension()==1 then
    transformParams = addOuterDim(_tranformParams)
  else
    transformParams = _tranformParams
  end

  self:check(transformParams)
  local batchSize = transformParams:size(1)

  if self.fullMode then
    self.output = transformParams:view(batchSize, 3, 4)
  else
    local completeTransformation = torch.zeros(batchSize,4,4):typeAs(transformParams)
    completeTransformation:select(3,1):select(2,1):add(1)
    completeTransformation:select(3,2):select(2,2):add(1)
    completeTransformation:select(3,3):select(2,3):add(1)
    completeTransformation:select(3,4):select(2,4):add(1)
        
    local idTransform = completeTransformation:clone()
    local transformationBuffer = torch.Tensor(batchSize,4,4):typeAs(transformParams)

    local paramIndex = 1
    -- not using rotation yet
    self.rotationOutput = idTransform:narrow(3,1,3):narrow(2,1,3):clone()
        
    if self.useScale then
      local scaleFactors = transformParams:select(2,paramIndex)
      paramIndex = paramIndex + 1

      transformationBuffer:zero()
      transformationBuffer:select(3,1):select(2,1):copy(scaleFactors)
      transformationBuffer:select(3,2):select(2,2):copy(scaleFactors)
      transformationBuffer:select(3,3):select(2,3):copy(scaleFactors)
      transformationBuffer:select(3,4):select(2,4):add(1)
      completeTransformation = torch.bmm(transformationBuffer, completeTransformation)
      self.scaleOutput = transformationBuffer:narrow(3,1,3):narrow(2,1,3):clone() -- we'll need this to implement rotation
    else
      self.scaleOutput = idTransform:narrow(3,1,3):narrow(2,1,3):clone()
    end
    
    if self.useTranslation then
      local txs = transformParams:select(2,paramIndex)
      local tys = transformParams:select(2,paramIndex+1)
      local tzs = transformParams:select(2,paramIndex+2)

      transformationBuffer:zero()
      transformationBuffer:select(3,1):select(2,1):add(1)
      transformationBuffer:select(3,2):select(2,2):add(1)
      transformationBuffer:select(3,3):select(2,3):add(1)
      transformationBuffer:select(3,4):select(2,4):add(1)
      transformationBuffer:select(3,4):select(2,1):copy(txs)
      transformationBuffer:select(3,4):select(2,2):copy(tys)
      transformationBuffer:select(3,4):select(2,3):copy(tzs)

      completeTransformation = torch.bmm(transformationBuffer, completeTransformation)
    end

    self.output=completeTransformation:narrow(2,1,3)
  end

  if _tranformParams:nDimension()==1 then
    self.output = self.output:select(1,1)
  end
  return self.output
end


function ATMG:updateGradInput(_tranformParams, _gradParams)
  local transformParams, gradParams
  if _tranformParams:nDimension()==1 then
    transformParams = addOuterDim(_tranformParams)
    gradParams = addOuterDim(_gradParams):clone()
  else
    transformParams = _tranformParams
    gradParams = _gradParams:clone()
  end

  local batchSize = transformParams:size(1)
  if self.fullMode then
    self.gradInput = gradParams:view(batchSize, 12)
  else
    local paramIndex = transformParams:size(2)
    self.gradInput:resizeAs(transformParams)
    if self.useTranslation then
      local gradInputTranslationParams = self.gradInput:narrow(2,paramIndex-2,3)
      paramIndex = paramIndex-3
      local selectedGradParams = gradParams:select(3,4)
      gradInputTranslationParams:copy(selectedGradParams)
    end

    if self.useScale then
      local gradInputScaleparams = self.gradInput:narrow(2,paramIndex,1)
      local sParams = transformParams:select(2,paramIndex)
      paramIndex = paramIndex-1

      local selectedOutput = self.rotationOutput
      local selectedGradParams = gradParams:narrow(2,1,3):narrow(3,1,3)
      gradInputScaleparams:copy(torch.cmul(selectedOutput, selectedGradParams):sum(2):sum(3))
    end

  end

  if _tranformParams:nDimension()==1 then
    self.gradInput = self.gradInput:select(1,1)
  end
  return self.gradInput
end


