#include "luaT.h"
#include "THC.h"

#include "utils.c"

#include "TrilinearSamplerBTHWC.cu"

LUA_EXTERNC DLL_EXPORT int luaopen_libcustn3d(lua_State *L);

int luaopen_libcustn3d(lua_State *L)
{
  lua_newtable(L);
  cunn_TrilinearSamplerBTHWC_init(L);

  return 1;
}
