#include <assert.h>
#include <cuda_runtime.h>
#include "OpenGlCudaHelper.h"
#include "CudaLifeFunctions.h"
#include <algorithm>

namespace mf {

	__global__ void naiveGOL(const ubyte* lifeData, uint worldWidth, uint worldHeight, ubyte* resultLifeData) {
		uint worldSize = worldWidth * worldHeight;

		for (uint cellId = blockIdx.x * blockDim.x + threadIdx.x;
				cellId < worldSize;
				cellId += blockDim.x * gridDim.x) {

			uint x = cellId % worldWidth;
			uint yAbs = cellId - x;

			uint xLeft = (x + worldWidth - 1) % worldWidth;
			uint xRight = (x + 1) % worldWidth;

			uint yAbsUp = (yAbs + worldSize - worldWidth) % worldSize;
			uint yAbsDown = (yAbs + worldWidth) % worldSize;

			// Count alive cells.
			uint aliveCells = lifeData[xLeft + yAbsUp] + lifeData[x + yAbsUp] + lifeData[xRight + yAbsUp]
				+ lifeData[xLeft + yAbs] + lifeData[xRight + yAbs]
				+ lifeData[xLeft + yAbsDown] + lifeData[x + yAbsDown] + lifeData[xRight + yAbsDown];

			resultLifeData[x + yAbs] = aliveCells == 3 || (aliveCells == 2 && lifeData[x + yAbs]) ? 1 : 0;
		}
	}

	/// Runs a kernel for simple byte-per-cell world evaluation.
	bool runNaiveGOL(ubyte*& d_lifeData, ubyte*& d_lifeDataBuffer, size_t worldWidth, size_t worldHeight,
			size_t iterationsCount, ushort threadsCount) {

		if ((worldWidth * worldHeight) % threadsCount != 0) {
			return false;
		}

		size_t reqBlocksCount = (worldWidth * worldHeight) / threadsCount;
		ushort blocksCount = (ushort)std::min((size_t)32768, reqBlocksCount);

		for (size_t i = 0; i < iterationsCount; ++i) {
			naiveGOL<<<blocksCount, threadsCount>>>(d_lifeData, uint(worldWidth), uint(worldHeight),
				d_lifeDataBuffer);
			std::swap(d_lifeData, d_lifeDataBuffer);
		}
		checkCudaErrors(cudaDeviceSynchronize());

		return true;
	}

	__global__ void visualizationKernel(const ubyte* lifeData, uint worldWidth, uint worldHeight, uchar4* destination,
			int destWidth, int detHeight, int2 displacement, double zoomFactor, int multisample, bool simulateColors,
			bool cyclic, bool bitLife) {

		uint pixelId = blockIdx.x * blockDim.x + threadIdx.x;

		int x = (int)floor(((int)(pixelId % destWidth) - displacement.x) * zoomFactor);
		int y = (int)floor(((int)(pixelId / destWidth) - displacement.y) * zoomFactor);

		if (cyclic) {
			x = ((x % (int)worldWidth) + worldWidth) % worldWidth;
			y = ((y % (int)worldHeight) + worldHeight) % worldHeight;
		}
		else if (x < 0 || y < 0 || x >= worldWidth || y >= worldHeight) {
			destination[pixelId].x = 127;
			destination[pixelId].y = 127;
			destination[pixelId].z = 127;
			return;
		}

		int value = 0; 
		int increment = 255 / (multisample * multisample);

		if (bitLife) {
			for (int dy = 0; dy < multisample; ++dy) {
				int yAbs = (y + dy) * worldWidth;
				for (int dx = 0; dx < multisample; ++dx) {
					int xBucket = yAbs + x + dx;
					value += ((lifeData[xBucket >> 3] >> (7 - (xBucket & 0x7))) & 0x1) * increment;
				}
			}
		}
		else {
			for (int dy = 0; dy < multisample; ++dy) {
				int yAbs = (y + dy) * worldWidth;
				for (int dx = 0; dx < multisample; ++dx) {
					value += lifeData[yAbs + (x + dx)] * increment;
				}
			}
		}

		bool isNotOnBoundary = !cyclic || !(x == 0 || y == 0);

		if (simulateColors) {
			if (value > 0) {
				if (destination[pixelId].w > 0) {
					// Stayed alive - get darker.
					if (destination[pixelId].y > 80) {
						if (isNotOnBoundary) {
							--destination[pixelId].x;
						}
						--destination[pixelId].y;
						--destination[pixelId].z;
					}
				}
				else {
					// Born - full white color.
					destination[pixelId].x = 255;
					destination[pixelId].y = 255;
					destination[pixelId].z = 255;
				}
			}
			else {
				if (destination[pixelId].w > 0) {
					// Died - dark red.
					if (isNotOnBoundary) {
						destination[pixelId].x = 128;
					}
					destination[pixelId].y = 0;
					destination[pixelId].z = 0;
				}
				else {
					// Stayed dead - get darker.
					if (destination[pixelId].x > 8) {
						if (isNotOnBoundary) {
						}
						destination[pixelId].x -= 8;
					}
					else {
						destination[pixelId].x = 0;
					}
				}
			}
		}
		else {
			destination[pixelId].x = value;
			destination[pixelId].y = isNotOnBoundary ? value : 255;
			destination[pixelId].z = value;
		}

		destination[pixelId].w = value;
	}

	/// Runs a kernel for rendering of life world on the screen.
	void runVisualization(const ubyte* d_lifeData, size_t worldWidth, size_t worldHeight, uchar4* destination,
			int destWidth, int destHeight, int displacementX, int displacementY, int zoom, bool simulateColors,
			bool cyclic, bool bitLife) {

		ushort threadsCount = 256;
		assert((worldWidth * worldHeight) % threadsCount == 0);
		size_t reqBlocksCount = (destWidth * destHeight) / threadsCount;
		assert(reqBlocksCount < 65536);
		ushort blocksCount = (ushort)reqBlocksCount;

		int multisample = std::min(4, (int)std::pow(2, std::max(0, zoom)));
		visualizationKernel<<<blocksCount, threadsCount>>>(d_lifeData, uint(worldWidth), uint(worldHeight), destination,
			destWidth, destHeight, make_int2(displacementX, displacementY), std::pow(2, zoom),
			multisample, zoom > 1 ? false : simulateColors, cyclic, bitLife);
		checkCudaErrors(cudaDeviceSynchronize());
	}

	__global__ void bitLifeEncodeKernel(const ubyte* lifeData, size_t encWorldSize, ubyte* resultEncodedLifeData) {

		for (size_t outputBucketId = blockIdx.x * blockDim.x + threadIdx.x;
				outputBucketId < encWorldSize;
				outputBucketId += blockDim.x * gridDim.x) {

			size_t cellId = outputBucketId << 3;

			ubyte result = lifeData[cellId] << 7 | lifeData[cellId + 1] << 6 | lifeData[cellId + 2] << 5
				| lifeData[cellId + 3] << 4 | lifeData[cellId + 4] << 3 | lifeData[cellId + 5] << 2
				| lifeData[cellId + 6] << 1 | lifeData[cellId + 7];

			resultEncodedLifeData[outputBucketId] = result;
		}

	}

	void runBitLifeEncodeKernel(const ubyte* d_lifeData, uint worldWidth, uint worldHeight, ubyte* d_encodedLife) {

		assert(worldWidth % 8 == 0);
		size_t worldEncDataWidth = worldWidth / 8;
		size_t encWorldSize = worldEncDataWidth * worldHeight;

		ushort threadsCount = 256;
		assert(encWorldSize % threadsCount == 0);
		size_t reqBlocksCount = encWorldSize / threadsCount;
		ushort blocksCount = (ushort)std::min((size_t)32768, reqBlocksCount);

		bitLifeEncodeKernel<<<blocksCount, threadsCount>>>(d_lifeData, encWorldSize, d_encodedLife);
		checkCudaErrors(cudaDeviceSynchronize());
	}

	__global__ void bitLifeDecodeKernel(const ubyte* encodedLifeData, uint encWorldSize, ubyte* resultDecodedlifeData) {

		for (uint outputBucketId = blockIdx.x * blockDim.x + threadIdx.x;
				outputBucketId < encWorldSize;
				outputBucketId += blockDim.x * gridDim.x) {

			uint cellId = outputBucketId << 3;
			ubyte dataBucket = encodedLifeData[outputBucketId];

			resultDecodedlifeData[cellId] = dataBucket >> 7;
			resultDecodedlifeData[cellId + 1] = (dataBucket >> 6) & 0x01;
			resultDecodedlifeData[cellId + 2] = (dataBucket >> 5) & 0x01;
			resultDecodedlifeData[cellId + 3] = (dataBucket >> 4) & 0x01;
			resultDecodedlifeData[cellId + 4] = (dataBucket >> 3) & 0x01;
			resultDecodedlifeData[cellId + 5] = (dataBucket >> 2) & 0x01;
			resultDecodedlifeData[cellId + 6] = (dataBucket >> 1) & 0x01;
			resultDecodedlifeData[cellId + 7] = dataBucket & 0x01;
		}

	}

	void runBitLifeDecodeKernel(const ubyte* d_encodedLife, uint worldWidth, uint worldHeight, ubyte* d_lifeData) {

		assert(worldWidth % 8 == 0);
		uint worldEncDataWidth = worldWidth / 8;
		uint encWorldSize = worldEncDataWidth * worldHeight;

		ushort threadsCount = 256;
		assert(encWorldSize % threadsCount == 0);
		uint reqBlocksCount = encWorldSize / threadsCount;
		ushort blocksCount = ushort(std::min(32768u, reqBlocksCount));
		bitLifeDecodeKernel<<<blocksCount, threadsCount>>>(d_encodedLife, encWorldSize, d_lifeData);
		checkCudaErrors(cudaDeviceSynchronize());
}

	__device__ inline uint getCellState(uint x, uint y, uint key) {
		uint index = y * 6 + x;
		return (key >> ((3 * 6 - 1) - index)) & 0x1;
	}

	__global__ void bitLifeKernelCounting(const ubyte* lifeData, uint worldDataWidth, uint worldHeight,
			uint bytesPerThread, ubyte* resultLifeData) {

		uint worldSize = (worldDataWidth * worldHeight);

		for (uint cellId = (blockIdx.x * blockDim.x + threadIdx.x) * bytesPerThread;
				cellId < worldSize;
				cellId += blockDim.x * gridDim.x * bytesPerThread) {

			uint x = (cellId + worldDataWidth - 1) % worldDataWidth;  // Start at block x - 1.
			uint yAbs = (cellId / worldDataWidth) * worldDataWidth;
			uint yAbsUp = (yAbs + worldSize - worldDataWidth) % worldSize;
			uint yAbsDown = (yAbs + worldDataWidth) % worldSize;

			// Initialize data with previous byte and current byte.
			uint data0 = (uint)lifeData[x + yAbsUp] << 16;
			uint data1 = (uint)lifeData[x + yAbs] << 16;
			uint data2 = (uint)lifeData[x + yAbsDown] << 16;

			x = (x + 1) % worldDataWidth;
			data0 |= (uint)lifeData[x + yAbsUp] << 8;
			data1 |= (uint)lifeData[x + yAbs] << 8;
			data2 |= (uint)lifeData[x + yAbsDown] << 8;

			for (uint i = 0; i < bytesPerThread; ++i) {
				uint oldX = x;  // Old x is referring to current center cell.
				x = (x + 1) % worldDataWidth;
				data0 |= (uint)lifeData[x + yAbsUp];
				data1 |= (uint)lifeData[x + yAbs];
				data2 |= (uint)lifeData[x + yAbsDown];

				uint result = 0;
				for (uint j = 0; j < 8; ++j) {
					uint aliveCells = (data0 & 0x14000) + (data1 & 0x14000) + (data2 & 0x14000);
					aliveCells >>= 14;
					aliveCells = (aliveCells & 0x3) + (aliveCells >> 2) + ((data0 >> 15) & 0x1u)
						+ ((data2 >> 15) & 0x1u);

					result = result << 1 | (aliveCells == 3 || (aliveCells == 2 && (data1 & 0x8000u)) ? 1u : 0u);

					data0 <<= 1;
					data1 <<= 1;
					data2 <<= 1;
				}

				resultLifeData[oldX + yAbs] = result;
			}
		}
	}

	__device__ inline uint swapEndianessUint32(uint val) {
		val = ((val << 8) & 0xFF00FF00u) | ((val >> 8) & 0xFF00FFu);
		return (val << 16) | ((val >> 16) & 0xFFFFu);
	}

	bool runBitLifeKernel(ubyte*& d_encodedLifeData, ubyte*& d_encodedlifeDataBuffer, 
			size_t worldWidth, size_t worldHeight, size_t iterationsCount, ushort threadsCount, uint bytesPerThread) {

		if (worldWidth % 8 != 0) {
			return false;
		}

		size_t worldEncDataWidth = worldWidth / 8;

		if (worldEncDataWidth % bytesPerThread != 0) {
			return false;
		}

		size_t encWorldSize = worldEncDataWidth * worldHeight;
		if (encWorldSize > std::numeric_limits<uint>::max()) {
			return false;
		}

		if ((encWorldSize / bytesPerThread) % threadsCount != 0) {
			return false;
		}

		size_t reqBlocksCount = (encWorldSize / bytesPerThread) / threadsCount;
		ushort blocksCount = ushort(std::min(size_t(32768), reqBlocksCount));

		for (size_t i = 0; i < iterationsCount; ++i) {
			bitLifeKernelCounting<<<blocksCount, threadsCount>>>(d_encodedLifeData, uint(worldEncDataWidth),
				uint(worldHeight), bytesPerThread, d_encodedlifeDataBuffer);
			std::swap(d_encodedLifeData, d_encodedlifeDataBuffer);
		}
		
		checkCudaErrors(cudaDeviceSynchronize());
		return true;
	}

}