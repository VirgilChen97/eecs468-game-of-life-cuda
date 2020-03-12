#pragma once

#include <limits>
#include <assert.h>
#include <stdint.h>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <time.h>
#include <chrono>

#include <GL/glew.h>
#include <GL/freeglut.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>


namespace mf {
	typedef glm::ivec2 Vector2i;
	typedef glm::ivec3 Vector3i;
	typedef glm::mediump_vec3 Vector3f;
	typedef glm::mediump_vec4 Vector4f;
	typedef glm::detail::tvec4<unsigned char> Vector4c;

	typedef unsigned char ubyte;
	typedef unsigned short ushort;
	typedef unsigned int uint;
}
