#pragma once
#include "OpenGlCudaHelper.h"
#include "CudaLifeFunctions.h"
#include "CpuLife.h"
#include <vector>

namespace mf {

	template<typename NoCppFileNeeded = int>
	class TGpuLife {

	private:
		ubyte* d_lifeData;
		ubyte* d_lifeDataBuffer;

		ubyte* d_encLifeData;
		ubyte* d_encLifeDataBuffer;

		ubyte* d_lookupTable;

		/// If unsuccessful allocation occurs, size is saved and never tried to allocate again to avoid many
		/// unsuccessful allocations in the row.
		size_t m_unsuccessAllocSize;


		/// Current width of world.
		size_t m_worldWidth;
		/// Current height of world.
		size_t m_worldHeight;

	public:
		TGpuLife()
			: d_lifeData(NULL)
			, d_lifeDataBuffer(NULL)
			, d_encLifeData(NULL)
			, d_encLifeDataBuffer(NULL)
			, d_lookupTable(NULL)
			, m_unsuccessAllocSize(std::numeric_limits<size_t>::max())
			, m_worldWidth(0)
			, m_worldHeight(0)
		{}

		~TGpuLife() {
			freeBuffers();
			checkCudaErrors(cudaFree(d_lookupTable));
			d_lookupTable = NULL;
		}

		const ubyte* getLifeData() const {
			return d_lifeData;
		}

		ubyte* lifeData() {
			return d_lifeData;
		}

		const ubyte* getBpcLifeData() const {
			return d_encLifeData;
		}

		ubyte* bpcLifeData() {
			return d_encLifeData;
		}

		bool areBuffersAllocated(bool bitLife) const {
			if (bitLife) {
				return d_encLifeData != NULL && d_encLifeDataBuffer != NULL;
			}
			else {
				return d_lifeData != NULL && d_lifeDataBuffer != NULL;
			}
		}

		bool allocBuffers(bool bitLife) {
			freeBuffers();

			if (bitLife) {
				size_t worldSize = (m_worldWidth / 8) * m_worldHeight;

				if (worldSize >= m_unsuccessAllocSize) {
					return false;
				}

				if (cudaMalloc(&d_encLifeData, worldSize) || cudaMalloc(&d_encLifeDataBuffer, worldSize)) {
					freeBuffers();
					m_unsuccessAllocSize = worldSize;
					return false;
				}
			}
			else {
				size_t worldSize = m_worldWidth * m_worldHeight;

				if (worldSize >= m_unsuccessAllocSize) {
					return false;
				}

				if (cudaMalloc(&d_lifeData, worldSize) || cudaMalloc(&d_lifeDataBuffer, worldSize)) {
					freeBuffers();
					m_unsuccessAllocSize = worldSize;
					return false;
				}
			}

			return true;
		}

		void freeBuffers() {
			checkCudaErrors(cudaFree(d_lifeData));
			d_lifeData = NULL;

			checkCudaErrors(cudaFree(d_lifeDataBuffer));
			d_lifeDataBuffer = NULL;

			checkCudaErrors(cudaFree(d_encLifeData));
			d_encLifeData = NULL;

			checkCudaErrors(cudaFree(d_encLifeDataBuffer));
			d_encLifeDataBuffer = NULL;
		}

		void resize(size_t newWidth, size_t newHeight) {
			freeBuffers();

			m_worldWidth = newWidth;
			m_worldHeight = newHeight;
		}

		void initThis(bool bitLife, CpuLife& cpuLife, size_t worldWidth, size_t worldHeight) {
			std::vector<ubyte> encData;

			if (bitLife) {
				size_t worldSize = (worldWidth / 8) * worldHeight;
				encData.resize(worldSize); 
				cpuLife.init(&encData[0], worldSize, 0xFF);
				checkCudaErrors(cudaMemcpy(d_encLifeData, &encData[0], worldSize, cudaMemcpyHostToDevice));
			}
			else {
				size_t worldSize = worldWidth * worldHeight;

				std::vector<ubyte> encData;
				encData.resize(worldSize); 
				cpuLife.init(&encData[0], worldSize, 0x1);
				checkCudaErrors(cudaMemcpy(d_lifeData, &encData[0], worldSize, cudaMemcpyHostToDevice));
			}
		}

		bool iterate(size_t lifeIteratinos, ushort threadsCount, bool bitLife,
			uint bitLifeBytesPerTrhead) {

			if (bitLife) {
				return runBitLifeKernel(d_encLifeData, d_encLifeDataBuffer, 
					m_worldWidth, m_worldHeight, lifeIteratinos, threadsCount, bitLifeBytesPerTrhead);
			}
			else {
				return runNaiveGOL(d_lifeData, d_lifeDataBuffer, m_worldWidth, m_worldHeight,
					lifeIteratinos, threadsCount);
			}
		}
	};
	typedef TGpuLife<> GpuLife;
}