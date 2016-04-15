#include "utils.h"
// Trilinear sampling is done in BTHWC (coalescing is not obvious in BCTHW)
// we assume BTHWC format in inputImages
// we assume BTHW(ZYX) format on grids

__device__ void getTopLeft(float x, int width, int& point, float& weight)
{
   /* for interpolation :
      stores in point and weight :
      - the x-coordinate of the pixel on the left (or y-coordinate of the upper pixel)
      - the weight for interpolating
   */

   float xcoord = (x + 1) * (width - 1) / 2;
   point = floor(xcoord);
   weight = 1 - (xcoord - point);
}

__device__ bool between(int value, int lowerBound, int upperBound)
{
   return (value >= lowerBound && value <= upperBound);
}

__global__ void trilinearSamplingFromGrid(float* inputImages_data, int inputImages_strideBatch, int inputImages_strideChannels, int inputImages_strideTime, int inputImages_strideHeight, int inputImages_strideWidth,
                                         float* grids_data, int grids_strideBatch, int grids_strideZYX, int grids_strideTime, int grids_strideHeight, int grids_strideWidth,
                                         float* output_data, int output_strideBatch, int output_strideChannels, int output_strideTime, int output_strideHeight, int output_strideWidth,
                                         int inputImages_channels, int inputImages_time, int inputImages_height, int inputImages_width)
{
   // each (32,16) block 16 output pixels (for coalescing the grid read)
   // x,y = coordinates (xOut = blockIdx.x*16+blockDim.y+threadIdx.y)
   // z = batch index
   // threadIdx.x : used for features (coalescing is trivial)
      
   const int xOut = blockIdx.x;
   const int yOut = blockIdx.y;
   const int tOut = threadIdx.x;
   const int b = blockIdx.z;
   
   const int width = inputImages_width;
   const int height = inputImages_height;
   const int frames = inputImages_time;
   
   float yf,xf,tf;

   float gridData[3];
   for (int coord = 0; coord <3; coord++)gridData[coord] = grids_data[b*grids_strideBatch + tOut*grids_strideTime + yOut*grids_strideHeight + xOut*grids_strideWidth + coord];

   tf = gridData[0];
   yf = gridData[1];
   xf = gridData[2];
   
   int tInTopLeft, yInTopLeft, xInTopLeft;
   float tWeightTopLeft, yWeightTopLeft, xWeightTopLeft;
   getTopLeft(xf, inputImages_width, xInTopLeft, xWeightTopLeft);
   getTopLeft(yf, inputImages_height, yInTopLeft, yWeightTopLeft);
   getTopLeft(tf, inputImages_time, tInTopLeft, tWeightTopLeft);
   
   const int outAddress = output_strideBatch * b + output_strideTime * tOut + output_strideHeight * yOut + output_strideWidth * xOut;
   const int in000Address = inputImages_strideBatch * b + inputImages_strideTime * tInTopLeft + inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft;
   const int in001Address = in000Address + inputImages_strideWidth;
   const int in010Address = in000Address + inputImages_strideHeight;
   const int in011Address = in010Address + inputImages_strideWidth;
   
   const int in100Address = in000Address + inputImages_strideTime;
   const int in101Address = in001Address + inputImages_strideTime;
   const int in110Address = in010Address + inputImages_strideTime;
   const int in111Address = in011Address + inputImages_strideTime;



   float v=0;
   float in000=0;
   float in001=0;
   float in010=0;
   float in011=0;
   float in100=0;
   float in101=0;
   float in110=0;
   float in111=0;

   bool IsIn000 = between(tInTopLeft, 0, frames-1) && between(yInTopLeft, 0, height-1) && between(xInTopLeft, 0, width-1);
   bool IsIn001 = between(tInTopLeft, 0, frames-1) && between(yInTopLeft, 0, height-1) && between(xInTopLeft+1, 0, width-1);
   bool IsIn010 = between(tInTopLeft, 0, frames-1) && between(yInTopLeft+1, 0, height-1) && between(xInTopLeft, 0, width-1);
   bool IsIn011 = between(tInTopLeft, 0, frames-1) && between(yInTopLeft+1, 0, height-1) && between(xInTopLeft+1, 0, width-1);
   bool IsIn100 = between(tInTopLeft+1, 0, frames-1) && between(yInTopLeft, 0, height-1) && between(xInTopLeft, 0, width-1);
   bool IsIn101 = between(tInTopLeft+1, 0, frames-1) && between(yInTopLeft, 0, height-1) && between(xInTopLeft+1, 0, width-1);
   bool IsIn110 = between(tInTopLeft+1, 0, frames-1) && between(yInTopLeft+1, 0, height-1) && between(xInTopLeft, 0, width-1);
   bool IsIn111 = between(tInTopLeft+1, 0, frames-1) && between(yInTopLeft+1, 0, height-1) && between(xInTopLeft+1, 0, width-1);

   // interpolation happens here
   for(int t=0; t<inputImages_channels; t++)
   {
      if(IsIn000) in000 = inputImages_data[in000Address + t];
      if(IsIn001) in001 = inputImages_data[in001Address + t];
      if(IsIn010) in010 = inputImages_data[in010Address + t];
      if(IsIn011) in011 = inputImages_data[in011Address + t];
      if(IsIn100) in100 = inputImages_data[in100Address + t];
      if(IsIn101) in101 = inputImages_data[in101Address + t];
      if(IsIn110) in110 = inputImages_data[in110Address + t];
      if(IsIn111) in111 = inputImages_data[in111Address + t];

      v = tWeightTopLeft * yWeightTopLeft * xWeightTopLeft * in000
        + tWeightTopLeft * yWeightTopLeft * (1 - xWeightTopLeft) * in001
        + tWeightTopLeft * (1 - yWeightTopLeft) * xWeightTopLeft * in010
        + tWeightTopLeft * (1 - yWeightTopLeft) * (1 - xWeightTopLeft) * in011
        + (1-tWeightTopLeft) * yWeightTopLeft * xWeightTopLeft * in100
        + (1-tWeightTopLeft) * yWeightTopLeft * (1 - xWeightTopLeft) * in101
        + (1-tWeightTopLeft) * (1 - yWeightTopLeft) * xWeightTopLeft * in110
        + (1-tWeightTopLeft) * (1 - yWeightTopLeft) * (1 - xWeightTopLeft) * in111;
      
      output_data[outAddress + t] = v;
   }
}


static int cunn_TrilinearSamplerBTHWC_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");


   dim3 blocks(output->size[3], output->size[2], output->size[0]); // Width X Height X BatchSize
   dim3 threads(output->size[1]); // nTime - each thread will handle all input_channels together

   /* assume BTHWC */
   trilinearSamplingFromGrid <<< blocks, threads, 0, THCState_getCurrentStream(state) >>> (THCudaTensor_data(state, inputImages), 
                                                      THCudaTensor_stride(state, inputImages, 0), 
                                                      THCudaTensor_stride(state, inputImages, 4), 
                                                      THCudaTensor_stride(state, inputImages, 1), 
                                                      THCudaTensor_stride(state, inputImages, 2), 
                                                      THCudaTensor_stride(state, inputImages, 3),
                                                      THCudaTensor_data(state, grids),  
                                                      THCudaTensor_stride(state, grids, 0), 
                                                      THCudaTensor_stride(state, grids, 4),
                                                      THCudaTensor_stride(state, grids, 1),
                                                      THCudaTensor_stride(state, grids, 2), 
                                                      THCudaTensor_stride(state, grids, 3),
                                                      THCudaTensor_data(state, output),  
                                                      THCudaTensor_stride(state, output, 0), 
                                                      THCudaTensor_stride(state, output, 4),
                                                      THCudaTensor_stride(state, output, 1),
                                                      THCudaTensor_stride(state, output, 2), 
                                                      THCudaTensor_stride(state, output, 3),
                                                      THCudaTensor_size(state, inputImages, 4),
                                                      THCudaTensor_size(state, inputImages, 1),
                                                      THCudaTensor_size(state, inputImages, 2), 
                                                      THCudaTensor_size(state, inputImages, 3));


  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in TrilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  return 1;
}

template<bool onlyGrid> __global__ void backwardTrilinearSampling(float* inputImages_data, int inputImages_strideBatch, int inputImages_strideChannels, int inputImages_strideTime, int inputImages_strideHeight, int inputImages_strideWidth,
                                         float* gradInputImages_data, int gradInputImages_strideBatch, int gradInputImages_strideChannels, int gradInputImages_strideTime, int gradInputImages_strideHeight, int gradInputImages_strideWidth,
                                         float* grids_data, int grids_strideBatch, int grids_strideZYX, int grids_strideTime, int grids_strideHeight, int grids_strideWidth,
                                         float* gradGrids_data, int gradGrids_strideBatch, int gradGrids_strideYXZ, int gradGrids_strideTime, int gradGrids_strideHeight, int gradGrids_strideWidth,
                                         float* gradOutput_data, int gradOutput_strideBatch, int gradOutput_strideChannels, int gradOutput_strideTime, int gradOutput_strideHeight, int gradOutput_strideWidth,
                                         int inputImages_channels, int inputImages_time, int inputImages_height, int inputImages_width)
{
    // each (32,16) block 16 output pixels (for coalescing the grid read)
    // x,y = coordinates
    // z = batch index
    // threads : used for features
      
    const int xOut = blockIdx.x;
    const int yOut = blockIdx.y;
    const int tOut = threadIdx.x;
    const int b = blockIdx.z;

    const int width = inputImages_width;
    const int height = inputImages_height;
    const int frames = inputImages_time;

    float yf,xf,tf;

    float gridData[3];
    for (int coord = 0; coord <3; coord++)gridData[coord] = grids_data[b*grids_strideBatch + tOut*grids_strideTime + yOut*grids_strideHeight + xOut*grids_strideWidth + coord];

    tf = gridData[0];
    yf = gridData[1];
    xf = gridData[2];
   
    int tInTopLeft, yInTopLeft, xInTopLeft;
    float tWeightTopLeft, yWeightTopLeft, xWeightTopLeft;
    getTopLeft(xf, inputImages_width, xInTopLeft, xWeightTopLeft);
    getTopLeft(yf, inputImages_height, yInTopLeft, yWeightTopLeft);
    getTopLeft(tf, inputImages_time, tInTopLeft, tWeightTopLeft);

    const int gradOutputAddress = gradOutput_strideBatch * b + gradOutput_strideTime * tOut + gradOutput_strideHeight * yOut + gradOutput_strideWidth * xOut;

    const int in000Address = inputImages_strideBatch * b + inputImages_strideTime * tInTopLeft + inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft;
    const int in001Address = in000Address + inputImages_strideWidth;
    const int in010Address = in000Address + inputImages_strideHeight;
    const int in011Address = in010Address + inputImages_strideWidth;

    const int in100Address = in000Address + inputImages_strideTime;
    const int in101Address = in001Address + inputImages_strideTime;
    const int in110Address = in010Address + inputImages_strideTime;
    const int in111Address = in011Address + inputImages_strideTime;

    const int gradInputImages000Address = gradInputImages_strideBatch * b + gradInputImages_strideTime * tOut + gradInputImages_strideHeight * yInTopLeft + gradInputImages_strideWidth * xInTopLeft;
    const int gradInputImages001Address = gradInputImages000Address + gradInputImages_strideWidth;
    const int gradInputImages010Address = gradInputImages000Address + gradInputImages_strideHeight;
    const int gradInputImages011Address = gradInputImages010Address + gradInputImages_strideWidth;
    
    const int gradInputImages100Address = gradInputImages000Address + gradInputImages_strideTime;
    const int gradInputImages101Address = gradInputImages001Address + gradInputImages_strideTime;
    const int gradInputImages110Address = gradInputImages010Address + gradInputImages_strideTime;
    const int gradInputImages111Address = gradInputImages011Address + gradInputImages_strideTime;


    float dotProduct000 = 0;
    float dotProduct001 = 0;
    float dotProduct010 = 0;
    float dotProduct011 = 0;
    float dotProduct100 = 0;
    float dotProduct101 = 0;
    float dotProduct110 = 0;
    float dotProduct111 = 0;

    bool IsIn000 = between(tInTopLeft, 0, frames-1) && between(yInTopLeft, 0, height-1) && between(xInTopLeft, 0, width-1);
    bool IsIn001 = between(tInTopLeft, 0, frames-1) && between(yInTopLeft, 0, height-1) && between(xInTopLeft+1, 0, width-1);
    bool IsIn010 = between(tInTopLeft, 0, frames-1) && between(yInTopLeft+1, 0, height-1) && between(xInTopLeft, 0, width-1);
    bool IsIn011 = between(tInTopLeft, 0, frames-1) && between(yInTopLeft+1, 0, height-1) && between(xInTopLeft+1, 0, width-1);
    bool IsIn100 = between(tInTopLeft+1, 0, frames-1) && between(yInTopLeft, 0, height-1) && between(xInTopLeft, 0, width-1);
    bool IsIn101 = between(tInTopLeft+1, 0, frames-1) && between(yInTopLeft, 0, height-1) && between(xInTopLeft+1, 0, width-1);
    bool IsIn110 = between(tInTopLeft+1, 0, frames-1) && between(yInTopLeft+1, 0, height-1) && between(xInTopLeft, 0, width-1);
    bool IsIn111 = between(tInTopLeft+1, 0, frames-1) && between(yInTopLeft+1, 0, height-1) && between(xInTopLeft+1, 0, width-1);

    /*
         In that loop we accumulate
         - gradients into the gradInputImages array with atomic adds
         - we compute the dot product that we need for the grid gradient
    */

    for(int t=threadIdx.x; t<inputImages_channels; t++)
    {
        float gradOutValue = gradOutput_data[gradOutputAddress + t];
        // bool between(int value, int lowerBound, int upperBound)
        if(IsIn000)
        {
            float in000 = inputImages_data[in000Address + t];
            dotProduct000 += in000 * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImages000Address + t], tWeightTopLeft * yWeightTopLeft * xWeightTopLeft * gradOutValue);
        }

        if(IsIn001)
        {
            float in001 = inputImages_data[in001Address + t];
            dotProduct001 += in001 * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImages001Address + t], tWeightTopLeft * yWeightTopLeft * (1-xWeightTopLeft) * gradOutValue);
        }

        if(IsIn010)
        {
            float in010 = inputImages_data[in010Address + t];
            dotProduct010 += in010 * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImages010Address + t], tWeightTopLeft * (1-yWeightTopLeft) * xWeightTopLeft * gradOutValue);
        }

        if(IsIn011)
        {
            float in011 = inputImages_data[in011Address + t];
            dotProduct011 += in011 * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImages011Address + t], tWeightTopLeft * (1-yWeightTopLeft) * (1-xWeightTopLeft) * gradOutValue);
        }
        
        if(IsIn100)
        {
            float in100 = inputImages_data[in100Address + t];
            dotProduct100 += in100 * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImages100Address + t], (1-tWeightTopLeft) * yWeightTopLeft * xWeightTopLeft * gradOutValue);
        }

        if(IsIn101)
        {
            float in101 = inputImages_data[in101Address + t];
            dotProduct101 += in101 * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImages101Address + t], (1-tWeightTopLeft) * yWeightTopLeft * (1-xWeightTopLeft) * gradOutValue);
        }

        if(IsIn110)
        {
            float in110 = inputImages_data[in110Address + t];
            dotProduct110 += in110 * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImages110Address + t], (1-tWeightTopLeft) * (1-yWeightTopLeft) * xWeightTopLeft * gradOutValue);
        }

        if(IsIn111)
        {
            float in111 = inputImages_data[in111Address + t];
            dotProduct111 += in111 * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImages111Address + t], (1-tWeightTopLeft) * (1-yWeightTopLeft) * (1-xWeightTopLeft) * gradOutValue);
        }
    }
    
    float xg = - tWeightTopLeft * yWeightTopLeft * dotProduct000
        + tWeightTopLeft * yWeightTopLeft * dotProduct001
        - tWeightTopLeft * (1 - yWeightTopLeft) * dotProduct010
        + tWeightTopLeft * (1 - yWeightTopLeft) * dotProduct011
        - (1-tWeightTopLeft) * yWeightTopLeft * dotProduct100
        + (1-tWeightTopLeft) * yWeightTopLeft * dotProduct101
        - (1-tWeightTopLeft) * (1 - yWeightTopLeft) * dotProduct110
        + (1-tWeightTopLeft) * (1 - yWeightTopLeft) * dotProduct111;
    
    float yg = - tWeightTopLeft * xWeightTopLeft * dotProduct000
        - tWeightTopLeft * (1 - xWeightTopLeft) * dotProduct001
        + tWeightTopLeft * xWeightTopLeft * dotProduct010
        + tWeightTopLeft * (1 - xWeightTopLeft) * dotProduct011
        - (1-tWeightTopLeft) * xWeightTopLeft * dotProduct100
        - (1-tWeightTopLeft) * (1 - xWeightTopLeft) * dotProduct101
        + (1-tWeightTopLeft) * xWeightTopLeft * dotProduct110
        + (1-tWeightTopLeft) * (1 - xWeightTopLeft) * dotProduct111;
    
    float zg = - yWeightTopLeft * xWeightTopLeft * dotProduct000
        - yWeightTopLeft * (1 - xWeightTopLeft) * dotProduct001
        - (1 - yWeightTopLeft) * xWeightTopLeft * dotProduct010
        - (1 - yWeightTopLeft) * (1 - xWeightTopLeft) * dotProduct011
        + yWeightTopLeft * xWeightTopLeft * dotProduct100
        + yWeightTopLeft * (1 - xWeightTopLeft) * dotProduct101
        + (1 - yWeightTopLeft) * xWeightTopLeft * dotProduct110
        + (1 - yWeightTopLeft) * (1 - xWeightTopLeft) * dotProduct111;
    
    float gridGrads[3];
    gridGrads[0] = zg * (inputImages_time-1) / 2;
    gridGrads[1] = yg * (inputImages_height-1) / 2;
    gridGrads[2] = xg * (inputImages_width-1) / 2;
    int coord;
    for(coord=0;coord<3;coord++) {gradGrids_data[b*gradGrids_strideBatch + tOut*gradGrids_strideTime + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth + coord] = gridGrads[coord];
    }
}

static int cunn_TrilinearSamplerBTHWC_updateGradInput(lua_State *L)
{
    THCState *state = getCutorchState(L);
    THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
    THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
    THCudaTensor *gradInputImages = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");
    THCudaTensor *gradGrids = (THCudaTensor *)luaT_checkudata(L, 5, "torch.CudaTensor");
    THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 6, "torch.CudaTensor");

    dim3 blocks(gradOutput->size[3], gradOutput->size[2], gradOutput->size[0]); // Width X Height X BatchSize
    dim3 threads(gradOutput->size[1]); // nTime - each thread will handle all input_channels together

    backwardTrilinearSampling <false> <<< blocks, threads, 0, THCState_getCurrentStream(state) >>> (
                                                      THCudaTensor_data(state, inputImages), 
                                                      THCudaTensor_stride(state, inputImages, 0),
                                                      THCudaTensor_stride(state, inputImages, 4),
                                                      THCudaTensor_stride(state, inputImages, 1),
                                                      THCudaTensor_stride(state, inputImages, 2),
                                                      THCudaTensor_stride(state, inputImages, 3),
                                                      THCudaTensor_data(state, gradInputImages), 
                                                      THCudaTensor_stride(state, gradInputImages, 0),
                                                      THCudaTensor_stride(state, gradInputImages, 4),
                                                      THCudaTensor_stride(state, gradInputImages, 1),
                                                      THCudaTensor_stride(state, gradInputImages, 2),
                                                      THCudaTensor_stride(state, gradInputImages, 3),
                                                      THCudaTensor_data(state, grids), 
                                                      THCudaTensor_stride(state, grids, 0),
                                                      THCudaTensor_stride(state, grids, 4),
                                                      THCudaTensor_stride(state, grids, 1),
                                                      THCudaTensor_stride(state, grids, 2),
                                                      THCudaTensor_stride(state, grids, 3),
                                                      THCudaTensor_data(state, gradGrids), 
                                                      THCudaTensor_stride(state, gradGrids, 0),
                                                      THCudaTensor_stride(state, gradGrids, 4),
                                                      THCudaTensor_stride(state, gradGrids, 1),
                                                      THCudaTensor_stride(state, gradGrids, 2),
                                                      THCudaTensor_stride(state, gradGrids, 3),
                                                      THCudaTensor_data(state, gradOutput), 
                                                      THCudaTensor_stride(state, gradOutput, 0),
                                                      THCudaTensor_stride(state, gradOutput, 4),
                                                      THCudaTensor_stride(state, gradOutput, 1),
                                                      THCudaTensor_stride(state, gradOutput, 2),
                                                      THCudaTensor_stride(state, gradOutput, 3),
                                                      THCudaTensor_size(state, inputImages, 4),
                                                      THCudaTensor_size(state, inputImages, 1),
                                                      THCudaTensor_size(state, inputImages, 2), 
                                                      THCudaTensor_size(state, inputImages, 3));



  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in TrilinearSampler.updateGradInput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  return 1;
}


static int cunn_TrilinearSamplerBTHWC_updateGradInputOnlyGrid(lua_State *L)
{
    THCState *state = getCutorchState(L);
    THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
    THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
    THCudaTensor *gradGrids = (THCudaTensor *)luaT_checkudata(L, 5, "torch.CudaTensor");
    THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 6, "torch.CudaTensor");

    dim3 blocks(gradOutput->size[3], gradOutput->size[2], gradOutput->size[0]); // Width X Height X BatchSize
    dim3 threads(gradOutput->size[1]); // nTime - each thread will handle all input_channels together

    backwardTrilinearSampling <true> <<< blocks, threads, 0, THCState_getCurrentStream(state) >>> (
                                                      THCudaTensor_data(state, inputImages), 
                                                      THCudaTensor_stride(state, inputImages, 0),
                                                      THCudaTensor_stride(state, inputImages, 4),
                                                      THCudaTensor_stride(state, inputImages, 1),
                                                      THCudaTensor_stride(state, inputImages, 2),
                                                      THCudaTensor_stride(state, inputImages, 3),
                                                      0, 
                                                      0,
                                                      0,
                                                      0,
                                                      0,
                                                      0,
                                                      THCudaTensor_data(state, grids), 
                                                      THCudaTensor_stride(state, grids, 0),
                                                      THCudaTensor_stride(state, grids, 4),
                                                      THCudaTensor_stride(state, grids, 1),
                                                      THCudaTensor_stride(state, grids, 2),
                                                      THCudaTensor_stride(state, grids, 3),
                                                      THCudaTensor_data(state, gradGrids), 
                                                      THCudaTensor_stride(state, gradGrids, 0),
                                                      THCudaTensor_stride(state, gradGrids, 4),
                                                      THCudaTensor_stride(state, gradGrids, 1),
                                                      THCudaTensor_stride(state, gradGrids, 2),
                                                      THCudaTensor_stride(state, gradGrids, 3),
                                                      THCudaTensor_data(state, gradOutput), 
                                                      THCudaTensor_stride(state, gradOutput, 0),
                                                      THCudaTensor_stride(state, gradOutput, 4),
                                                      THCudaTensor_stride(state, gradOutput, 1),
                                                      THCudaTensor_stride(state, gradOutput, 2),
                                                      THCudaTensor_stride(state, gradOutput, 3),
                                                      THCudaTensor_size(state, inputImages, 4),
                                                      THCudaTensor_size(state, inputImages, 1),
                                                      THCudaTensor_size(state, inputImages, 2), 
                                                      THCudaTensor_size(state, inputImages, 3));

    // check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in TrilinearSampler.updateGradInput: %s\n", cudaGetErrorString(err));
        THError("aborting");
    }
    return 1;
}

static const struct luaL_Reg cunn_TrilinearSamplerBTHWC__ [] = {
  {"TrilinearSamplerBTHWC_updateOutput", cunn_TrilinearSamplerBTHWC_updateOutput},
  {"TrilinearSamplerBTHWC_updateGradInput", cunn_TrilinearSamplerBTHWC_updateGradInput},
  {"TrilinearSamplerBTHWC_updateGradInputOnlyGrid", cunn_TrilinearSamplerBTHWC_updateGradInputOnlyGrid},
  {NULL, NULL}
};

static void cunn_TrilinearSamplerBTHWC_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_TrilinearSamplerBTHWC__, "nn");
  lua_pop(L,1);
}