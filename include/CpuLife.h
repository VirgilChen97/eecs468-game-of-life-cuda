#pragma once
#include <cstring>
namespace mf {

	template<typename NoCppFileNeeded = int>
	class TCpuLife {

	protected:
		//byte-per-pixel data.
		ubyte* m_data;
		ubyte* m_resultData;

		///bit-per-pixel data.
		ubyte* m_bpcData;
		ubyte* m_bpcResultData;

		size_t m_worldWidth;
		size_t m_worldHeight;
		size_t m_dataLength;


	public:
		TCpuLife()
				: m_data(NULL)
				, m_resultData(NULL)
				, m_bpcData(NULL)
				, m_bpcResultData(NULL)
				, m_worldWidth(0)
				, m_worldHeight(0)
				, m_dataLength(0) {
			srand(uint(time(NULL)));
		}

		~TCpuLife() {
			freeBuffers();
		}

		size_t getWorldWidth() const {
			return m_worldWidth;
		}

		size_t getWorldHeight() const {
			return m_worldHeight;
		}

		const ubyte* getLifeData() const {
			return m_data;
		}

		ubyte* lifeData() {
			return m_data;
		}

		const ubyte* getBpcLifeData() const {
			return m_bpcData;
		}

		bool areBuffersAllocated(bool bitLife) const {
			if (bitLife) {
				return m_bpcData != NULL && m_bpcResultData != NULL;
			}
			else {
				return m_data != NULL && m_resultData != NULL && m_bpcData != NULL;
			}
		}

		void freeBuffers() {
			delete[] m_data;
			m_data = NULL;

			delete[] m_resultData;
			m_resultData = NULL;

			delete[] m_bpcData;
			m_bpcData = NULL;

			delete[] m_bpcResultData;
			m_bpcResultData = NULL;

			m_dataLength = 0;
		}

		bool allocBuffers(bool bitLife) {
			freeBuffers();

			size_t dataLength = m_worldWidth * m_worldHeight;
			assert(dataLength % 8 == 0);
			size_t bitDataLength = dataLength / 8;

			try {
				// Bit-per-cell buffer is always needed for display.
				m_bpcData = new ubyte[bitDataLength];

				if (bitLife) {
					m_dataLength = bitDataLength;
					m_bpcResultData = new ubyte[m_dataLength];
				}
				else {
					m_dataLength = dataLength;
					m_data = new ubyte[m_dataLength];
					m_resultData = new ubyte[m_dataLength];
				}
			}
			catch (std::bad_alloc&) {
				freeBuffers();
				return false;
			}

			return true;
		}

		void resize(size_t newWidth, size_t newHeight) {
			freeBuffers();
			m_worldWidth = newWidth;
			m_worldHeight = newHeight;
		}

		void initThis() {
			if (m_data != NULL) {
				// Normal life.
				init(m_data, m_dataLength, 0x1u);
			}
			else {
				// Bit life.
				init(m_bpcData, m_dataLength, 0xFFu);
			}
		}

		void init(ubyte* data, size_t length, uint mask) {
			for (size_t i = 0; i < length; ++i) {
				data[i] = ubyte(rand() & mask);
			}
		}

		void encodeDataToBpc() {
			assert(m_data != NULL);

			ubyte* data = m_data;
			ubyte* encData = m_bpcData;
			size_t dataLength = m_worldWidth *  m_worldHeight;
			size_t encDataLength = dataLength / 8;
			std::memset(encData, 0, encDataLength);
			for (size_t i = 0; i < dataLength; ++i) {
				encData[i / 8] |= data[i] << (7 - (i % 8));
			}
		}

		// Evaluates life 
		bool iterate(size_t lifeIteratinos) {
			return iterateSerial(lifeIteratinos);
		}

		bool iterateSerial(size_t iterations) {
			for (size_t i = 0; i < iterations; ++i) {
				for (size_t y = 0; y < m_worldHeight; ++y) {
					size_t y0 = ((y + m_worldHeight - 1) % m_worldHeight) * m_worldWidth;
					size_t y1 = y * m_worldWidth;
					size_t y2 = ((y + 1) % m_worldHeight) * m_worldWidth;

					for (size_t x = 0; x < m_worldWidth; ++x) {
						size_t x0 = (x + m_worldWidth - 1) % m_worldWidth;
						size_t x2 = (x + 1) % m_worldWidth;

						ubyte aliveCells = countAliveCells(m_data, x0, x, x2, y0, y1, y2);
						m_resultData[y1 + x] = aliveCells == 3 || (aliveCells == 2 && m_data[x + y1]) ? 1 : 0;
					}
				}

				std::swap(m_data, m_resultData);
			}

			return true;
		}

	private:
		// Static variables for static function usage only.
		static ubyte* s_data;
		static ubyte* s_resultData;
		static size_t s_worldWidth;
		static size_t s_dataLength;

		/// Counts alive cells in given data on given coords.
		/// Y-coordinates y0, y1 and y2 are already pre-multiplied with world width.
		static inline ubyte countAliveCells(ubyte* data, size_t x0, size_t x1, size_t x2, size_t y0, size_t y1,
				size_t y2) {

			return data[x0 + y0] + data[x1 + y0] + data[x2 + y0]
				+ data[x0 + y1] + data[x2 + y1]
				+ data[x0 + y2] + data[x1 + y2] + data[x2 + y2];
		}
	};

	template <typename T> ubyte* TCpuLife<T>::s_data;
	template <typename T> ubyte* TCpuLife<T>::s_resultData;
	template <typename T> size_t TCpuLife<T>::s_worldWidth;
	template <typename T> size_t TCpuLife<T>::s_dataLength;

	typedef TCpuLife<> CpuLife;

}
