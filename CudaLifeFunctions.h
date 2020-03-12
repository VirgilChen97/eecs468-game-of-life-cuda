#pragma once

namespace mf {

	typedef unsigned char ubyte;
	typedef unsigned short ushort;
	typedef unsigned int uint;


	extern "C" bool runNaiveGOL(ubyte*& d_lifeData, ubyte*& d_lifeDataBuffer, size_t worldWidth,
		size_t worldHeight, size_t iterationsCount, ushort threadsCount);

	extern "C" void runBitLifeEncodeKernel(const ubyte* d_lifeData, uint worldWidth, uint worldHeight,
		ubyte* d_encodedLife);

	extern "C" void runBitLifeDecodeKernel(const ubyte* d_encodedLife, uint worldWidth, uint worldHeight,
		ubyte* d_lifeData);

	extern "C" bool runBitLifeKernel(ubyte*& d_lifeData, ubyte*& d_lifeDataBuffer ,
		size_t worldWidth, size_t worldHeight, size_t iterationsCount, ushort threadsCount, uint bytesPerThread);


	extern "C" void runVisualization(const ubyte* d_lifeData, size_t worldWidth, size_t worldHeight,
		uchar4 *destination, int destWidth, int detHeight, int displacementX, int displacementY, int zoom,
		bool simulateColors, bool cyclic, bool bitLife);

}