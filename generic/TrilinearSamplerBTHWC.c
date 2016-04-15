#ifndef TH3d_GENERIC_FILE
#define TH3d_GENERIC_FILE "generic/TrilinearSamplerBTHWC.c"
#else

#include <stdbool.h>

bool between(int value, int lowerBound, int upperBound)
{
   return (value >= lowerBound && value <= upperBound);
}

static int nn_(TrilinearSamplerBTHWC_updateOutput)(lua_State *L)
{
  THTensor *inputImages = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *grids = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  int batchsize = inputImages->size[0];
  int inputImages_time = inputImages->size[1];
  int inputImages_height = inputImages->size[2];
  int inputImages_width = inputImages->size[3];
  int output_time = output->size[1];
  int output_height = output->size[2];
  int output_width = output->size[3];
  int inputImages_channels = inputImages->size[4];

  int output_strideBatch = output->stride[0];
  int output_strideTime = output->stride[1];
  int output_strideHeight = output->stride[2];
  int output_strideWidth = output->stride[3];

  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_strideTime = inputImages->stride[1];
  int inputImages_strideHeight = inputImages->stride[2];
  int inputImages_strideWidth = inputImages->stride[3];

  int grids_strideBatch = grids->stride[0];
  int grids_strideTime = grids->stride[1];
  int grids_strideHeight = grids->stride[2];
  int grids_strideWidth = grids->stride[3];

  real *inputImages_data, *output_data, *grids_data;
  inputImages_data = THTensor_(data)(inputImages);
  output_data = THTensor_(data)(output);
  grids_data = THTensor_(data)(grids);

  int b, yOut, xOut, tOut;

  for(b=0; b < batchsize; b++)
  {
    for(tOut=0; tOut < output_time; tOut++)
    {
      for(yOut=0; yOut < output_height; yOut++)
      {
        for(xOut=0; xOut < output_width; xOut++)
        {
          //read the grid
          real tf = grids_data[b*grids_strideBatch + tOut*grids_strideTime + yOut*grids_strideHeight + xOut*grids_strideWidth+0];
          real yf = grids_data[b*grids_strideBatch + tOut*grids_strideTime + yOut*grids_strideHeight + xOut*grids_strideWidth+1];
          real xf = grids_data[b*grids_strideBatch + tOut*grids_strideTime + yOut*grids_strideHeight + xOut*grids_strideWidth+2];

          // get the weights for interpolation
          int yInTopLeft, xInTopLeft, tInTopLeft;
          real tWeightTopLeft, yWeightTopLeft, xWeightTopLeft;
 
          real xcoord = (xf + 1) * (inputImages_width - 1) / 2;
          xInTopLeft = floor(xcoord);
          xWeightTopLeft = 1 - (xcoord - xInTopLeft);

          real ycoord = (yf + 1) * (inputImages_height - 1) / 2;
          yInTopLeft = floor(ycoord);
          yWeightTopLeft = 1 - (ycoord - yInTopLeft);

          real tcoord = (tf + 1) * (inputImages_time - 1) / 2;
          tInTopLeft = floor(tcoord);
          tWeightTopLeft = 1 - (tcoord - tInTopLeft);

          const int outAddress = output_strideBatch * b + output_strideTime * tOut + output_strideHeight * yOut + output_strideWidth * xOut;
          const int in000Address = inputImages_strideBatch * b + inputImages_strideTime * tInTopLeft + inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft;
          const int in001Address = in000Address + inputImages_strideWidth;
          const int in010Address = in000Address + inputImages_strideHeight;
          const int in011Address = in010Address + inputImages_strideWidth;
   
          const int in100Address = in000Address + inputImages_strideTime;
          const int in101Address = in001Address + inputImages_strideTime;
          const int in110Address = in010Address + inputImages_strideTime;
          const int in111Address = in011Address + inputImages_strideTime;

          real v=0;
          real in000=0;
          real in001=0;
          real in010=0;
          real in011=0;
          real in100=0;
          real in101=0;
          real in110=0;
          real in111=0;

          // we are careful with the boundaries
          bool IsIn000 = between(tInTopLeft, 0, output_time-1) && between(yInTopLeft, 0, output_height-1) && between(xInTopLeft, 0, output_width-1);
          bool IsIn001 = between(tInTopLeft, 0, output_time-1) && between(yInTopLeft, 0, output_height-1) && between(xInTopLeft+1, 0, output_width-1);
          bool IsIn010 = between(tInTopLeft, 0, output_time-1) && between(yInTopLeft+1, 0, output_height-1) && between(xInTopLeft, 0, output_width-1);
          bool IsIn011 = between(tInTopLeft, 0, output_time-1) && between(yInTopLeft+1, 0, output_height-1) && between(xInTopLeft+1, 0, output_width-1);
          bool IsIn100 = between(tInTopLeft+1, 0, output_time-1) && between(yInTopLeft, 0, output_height-1) && between(xInTopLeft, 0, output_width-1);
          bool IsIn101 = between(tInTopLeft+1, 0, output_time-1) && between(yInTopLeft, 0, output_height-1) && between(xInTopLeft+1, 0, output_width-1);
          bool IsIn110 = between(tInTopLeft+1, 0, output_time-1) && between(yInTopLeft+1, 0, output_height-1) && between(xInTopLeft, 0, output_width-1);
          bool IsIn111 = between(tInTopLeft+1, 0, output_time-1) && between(yInTopLeft+1, 0, output_height-1) && between(xInTopLeft+1, 0, output_width-1);

          int t;
          // interpolation happens here
          for(t=0; t<inputImages_channels; t++)
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
      }
    }
  }

  return 1;
}

static int nn_(TrilinearSamplerBTHWC_updateGradInput)(lua_State *L)
{
  THTensor *inputImages = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *grids = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *gradInputImages = luaT_checkudata(L, 4, torch_Tensor);
  THTensor *gradGrids = luaT_checkudata(L, 5, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 6, torch_Tensor);

  bool onlyGrid=false;

  int batchsize = inputImages->size[0];
  int inputImages_time = inputImages->size[1];
  int inputImages_height = inputImages->size[2];
  int inputImages_width = inputImages->size[3];
  int output_time = output->size[1];
  int output_height = output->size[2];
  int output_width = output->size[3];
  int inputImages_channels = inputImages->size[4];

  int gradOutput_strideBatch = gradOutput->stride[0];
  int gradOutput_strideTime = gradOutput->stride[1];
  int gradOutput_strideHeight = gradOutput->stride[2];
  int gradOutput_strideWidth = gradOutput->stride[3];

  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_strideTime = inputImages->stride[1];
  int inputImages_strideHeight = inputImages->stride[2];
  int inputImages_strideWidth = inputImages->stride[3];

  int gradInputImages_strideBatch = gradInputImages->stride[0];
  int gradInputImages_strideTime = gradInputImages->stride[1];
  int gradInputImages_strideHeight = gradInputImages->stride[2];
  int gradInputImages_strideWidth = gradInputImages->stride[3];

  int grids_strideBatch = grids->stride[0];
  int grids_strideTime = grids->stride[1];
  int grids_strideHeight = grids->stride[2];
  int grids_strideWidth = grids->stride[3];

  int gradGrids_strideBatch = gradGrids->stride[0];
  int gradGrids_strideTime = gradGrids->stride[1];
  int gradGrids_strideHeight = gradGrids->stride[2];
  int gradGrids_strideWidth = gradGrids->stride[3];

  real *inputImages_data, *gradOutput_data, *grids_data, *gradGrids_data, *gradInputImages_data;
  inputImages_data = THTensor_(data)(inputImages);
  gradOutput_data = THTensor_(data)(gradOutput);
  grids_data = THTensor_(data)(grids);
  gradGrids_data = THTensor_(data)(gradGrids);
  gradInputImages_data = THTensor_(data)(gradInputImages);

  int b, tOut, yOut, xOut;

  for(b=0; b < batchsize; b++)
  {
    for(tOut=0; tOut < gradOutput_time; tOut++)
    {
        for(yOut=0; yOut < gradOutput_height; yOut++)
        {
          for(xOut=0; xOut < gradOutput_width; xOut++)
          {
            //read the grid
            real tf = grids_data[b*grids_strideBatch + tOut*grids_strideTime + yOut*grids_strideHeight + xOut*grids_strideWidth+0];
            real yf = grids_data[b*grids_strideBatch + tOut*grids_strideTime + yOut*grids_strideHeight + xOut*grids_strideWidth+1];
            real xf = grids_data[b*grids_strideBatch + tOut*grids_strideTime + yOut*grids_strideHeight + xOut*grids_strideWidth+2];

          // get the weights for interpolation
            int yInTopLeft, xInTopLeft, tInTopLeft;
            real tWeightTopLeft, yWeightTopLeft, xWeightTopLeft;
 
            real xcoord = (xf + 1) * (inputImages_width - 1) / 2;
            xInTopLeft = floor(xcoord);
            xWeightTopLeft = 1 - (xcoord - xInTopLeft);

            real ycoord = (yf + 1) * (inputImages_height - 1) / 2;
            yInTopLeft = floor(ycoord);
            yWeightTopLeft = 1 - (ycoord - yInTopLeft);

            real tcoord = (tf + 1) * (inputImages_time - 1) / 2;
            tInTopLeft = floor(tcoord);
            tWeightTopLeft = 1 - (tcoord - tInTopLeft);

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

            real dotProduct000 = 0;
            real dotProduct001 = 0;
            real dotProduct010 = 0;
            real dotProduct011 = 0;
            real dotProduct100 = 0;
            real dotProduct101 = 0;
            real dotProduct110 = 0;
            real dotProduct111 = 0;

            bool IsIn000 = between(tInTopLeft, 0, output_time-1) && between(yInTopLeft, 0, output_height-1) && between(xInTopLeft, 0, output_width-1);
            bool IsIn001 = between(tInTopLeft, 0, output_time-1) && between(yInTopLeft, 0, output_height-1) && between(xInTopLeft+1, 0, output_width-1);
            bool IsIn010 = between(tInTopLeft, 0, output_time-1) && between(yInTopLeft+1, 0, output_height-1) && between(xInTopLeft, 0, output_width-1);
            bool IsIn011 = between(tInTopLeft, 0, output_time-1) && between(yInTopLeft+1, 0, output_height-1) && between(xInTopLeft+1, 0, output_width-1);
            bool IsIn100 = between(tInTopLeft+1, 0, output_time-1) && between(yInTopLeft, 0, output_height-1) && between(xInTopLeft, 0, output_width-1);
            bool IsIn101 = between(tInTopLeft+1, 0, output_time-1) && between(yInTopLeft, 0, output_height-1) && between(xInTopLeft+1, 0, output_width-1);
            bool IsIn110 = between(tInTopLeft+1, 0, output_time-1) && between(yInTopLeft+1, 0, output_height-1) && between(xInTopLeft, 0, output_width-1);
            bool IsIn111 = between(tInTopLeft+1, 0, output_time-1) && between(yInTopLeft+1, 0, output_height-1) && between(xInTopLeft+1, 0, output_width-1);

            real v=0;
            real in000=0;
            real in001=0;
            real in010=0;
            real in011=0;
            real in100=0;
            real in101=0;
            real in110=0;
            real in111=0;

            int t;

            for(t=0; t<inputImages_channels; t++)
            {
                real gradOutValue = gradOutput_data[gradOutputAddress + t];
                if(IsIn000)
                {
                    real in000 = inputImages_data[in000Address + t];
                    dotProduct000 += in000 * gradOutValue;
                    if(!onlyGrid) gradInputImages_data[gradInputImages000Address + t] += tWeightTopLeft * yWeightTopLeft * xWeightTopLeft * gradOutValue;
                }

                if(IsIn001)
                {
                    real in001 = inputImages_data[in001Address + t];
                    dotProduct001 += in001 * gradOutValue;
                    if(!onlyGrid) gradInputImages_data[gradInputImages001Address + t] += tWeightTopLeft * yWeightTopLeft * (1-xWeightTopLeft) * gradOutValue;
                }

                if(IsIn010)
                {
                    real in010 = inputImages_data[in010Address + t];
                    dotProduct010 += in010 * gradOutValue;
                    if(!onlyGrid) gradInputImages_data[gradInputImages010Address + t] += tWeightTopLeft * (1-yWeightTopLeft) * xWeightTopLeft * gradOutValue;
                }

                if(IsIn011)
                {
                    real in011 = inputImages_data[in011Address + t];
                    dotProduct011 += in011 * gradOutValue;
                    if(!onlyGrid) gradInputImages_data[gradInputImages011Address + t] += tWeightTopLeft * (1-yWeightTopLeft) * (1-xWeightTopLeft) * gradOutValue;
                }

                if(IsIn100)
                {
                    real in100 = inputImages_data[in100Address + t];
                    dotProduct100 += in100 * gradOutValue;
                    if(!onlyGrid) gradInputImages_data[gradInputImages100Address + t] += (1-tWeightTopLeft) * yWeightTopLeft * xWeightTopLeft * gradOutValue;
                }

                if(IsIn101)
                {
                    real in101 = inputImages_data[in101Address + t];
                    dotProduct101 += in101 * gradOutValue;
                    if(!onlyGrid) gradInputImages_data[gradInputImages101Address + t] += (1-tWeightTopLeft) * yWeightTopLeft * (1-xWeightTopLeft) * gradOutValue;
                }

                if(IsIn110)
                {
                    real in110 = inputImages_data[in110Address + t];
                    dotProduct110 += in110 * gradOutValue;
                    if(!onlyGrid) gradInputImages_data[gradInputImages110Address + t] += (1-tWeightTopLeft) * (1-yWeightTopLeft) * xWeightTopLeft * gradOutValue);
                }

                if(IsIn111)
                {
                    real in111 = inputImages_data[in111Address + t];
                    dotProduct111 += in111 * gradOutValue;
                    if(!onlyGrid) gradInputImages_data[gradInputImages111Address + t] += (1-tWeightTopLeft) * (1-yWeightTopLeft) * (1-xWeightTopLeft) * gradOutValue);
                }
            }

            xf = - tWeightTopLeft * yWeightTopLeft * dotProduct000
                + tWeightTopLeft * yWeightTopLeft * dotProduct001
                - tWeightTopLeft * (1 - yWeightTopLeft) * dotProduct010
                + tWeightTopLeft * (1 - yWeightTopLeft) * dotProduct011
                - (1-tWeightTopLeft) * yWeightTopLeft * dotProduct100
                + (1-tWeightTopLeft) * yWeightTopLeft * dotProduct101
                - (1-tWeightTopLeft) * (1 - yWeightTopLeft) * dotProduct110
                + (1-tWeightTopLeft) * (1 - yWeightTopLeft) * dotProduct111;

            yf = - tWeightTopLeft * xWeightTopLeft * dotProduct000
                - tWeightTopLeft * (1 - xWeightTopLeft) * dotProduct001
                + tWeightTopLeft * xWeightTopLeft * dotProduct010
                + tWeightTopLeft * (1 - xWeightTopLeft) * dotProduct011
                - (1-tWeightTopLeft) * xWeightTopLeft * dotProduct100
                - (1-tWeightTopLeft) * (1 - xWeightTopLeft) * dotProduct101
                + (1-tWeightTopLeft) * xWeightTopLeft * dotProduct110
                + (1-tWeightTopLeft) * (1 - xWeightTopLeft) * dotProduct111;

            zf = - yWeightTopLeft * xWeightTopLeft * dotProduct000
                - yWeightTopLeft * (1 - xWeightTopLeft) * dotProduct001
                - (1 - yWeightTopLeft) * xWeightTopLeft * dotProduct010
                - (1 - yWeightTopLeft) * (1 - xWeightTopLeft) * dotProduct011
                + yWeightTopLeft * xWeightTopLeft * dotProduct100
                + yWeightTopLeft * (1 - xWeightTopLeft) * dotProduct101
                + (1 - yWeightTopLeft) * xWeightTopLeft * dotProduct110
                + (1 - yWeightTopLeft) * (1 - xWeightTopLeft) * dotProduct111;

            real gridGrads[3];
            gridGrads[0] = zf * (inputImages_time-1) / 2;
            gridGrads[1] = yf * (inputImages_height-1) / 2;
            gridGrads[2] = xf * (inputImages_width-1) / 2;

            for(int coord=0;coord<3;dim++) {gradGrids_data[b*gradGrids_strideBatch + tOut*gradGrids_strideTime + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth + d] = gridGrads[coord];
            }

          }
        }
    }
  }

  return 1;
}

static const struct luaL_Reg nn_(TrilinearSamplerBTHWC__) [] = {
  {"TrilinearSamplerBTHWC_updateOutput", nn_(TrilinearSamplerBTHWC_updateOutput)},
  {"TrilinearSamplerBTHWC_updateGradInput", nn_(TrilinearSamplerBTHWC_updateGradInput)},
  {NULL, NULL}
};

static void nn_(TrilinearSamplerBTHWC_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(TrilinearSamplerBTHWC__), "nn");
  lua_pop(L,1);
}

#endif
