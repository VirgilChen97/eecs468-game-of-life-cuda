#include "stdafx.h"
#include "OpenGlCudaHelper.h"
#include "CpuLife.h"
#include "GpuLife.h"
#include "CudaLifeFunctions.h"

namespace mf {

	int cudaDeviceId = -1;

	int screenWidth = 1024;
	int screenHeight = 768;
	bool updateTextureNextFrame = true;
	bool resizeTextureNextFrame = true;

	Vector2i mousePosition;
	int mouseButtons = 0;

	bool runGpuLife = false;
	bool lifeRunning = false;
	bool bitLife = false;
	ushort threadsCount = 256;

	bool resizeWorld = false;
	bool resetWorldPostprocessDisplay = true;

	// camera
	int zoom = -2;
	Vector2i translate = Vector2i(0, 0);
	ubyte* textureData = nullptr;
	size_t textureWidth = screenWidth;
	size_t textureHeight = screenHeight;
	bool cyclicWorld = false;

	// game of life settings
	uint bitLifeBytesPerTrhead = 1u;

	size_t worldWidth = 256;
	size_t worldHeight = 256;

	size_t newWorldWidth = 256;
	size_t newWorldHeight = 256;

	CpuLife cpuLife;
	GpuLife gpuLife;

	ubyte* d_cpuDisplayData = nullptr;


	/// Host-side texture pointer.
	uchar4* h_textureBufferData = nullptr;
	/// Device-side texture pointer.
	uchar4* d_textureBufferData = nullptr;

	GLuint gl_pixelBufferObject = 0;
	GLuint gl_texturePtr = 0;
	cudaGraphicsResource* cudaPboResource = nullptr;

	void freeAllBuffers() {
		cpuLife.freeBuffers();
		gpuLife.freeBuffers();

		checkCudaErrors(cudaFree(d_cpuDisplayData));
		d_cpuDisplayData = nullptr;
	}


	void resizeLifeWorld(size_t newWorldWidth, size_t newWorldHeight) {
		freeAllBuffers();

		worldWidth = newWorldWidth;
		worldHeight = newWorldHeight;

		cpuLife.resize(worldWidth, worldHeight);
		gpuLife.resize(worldWidth, worldHeight);
	}

	void initWorld(bool gpu, bool bitLife) {
		if (gpu) {
			gpuLife.initThis(bitLife, cpuLife);
		}
		else {
			cpuLife.initThis();  // Potential bad_alloc.
		}
	}

	void displayLife() {
		checkCudaErrors(cudaGraphicsMapResources(1, &cudaPboResource, 0));
		size_t num_bytes;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_textureBufferData, &num_bytes, cudaPboResource));

		bool ppc = !resetWorldPostprocessDisplay;

		if (gpuLife.getLifeData() != nullptr) {
			assert(gpuLife.getBpcLifeData() == nullptr);
			assert(d_cpuDisplayData == nullptr);
			runVisualization(gpuLife.getLifeData(), worldWidth, worldHeight, d_textureBufferData, screenWidth, screenHeight,
				translate.x, translate.y, zoom, ppc, cyclicWorld, false);
		}
		else if (gpuLife.getBpcLifeData() != nullptr) {
			assert(d_cpuDisplayData == nullptr);
			runVisualization(gpuLife.getBpcLifeData(), worldWidth, worldHeight, d_textureBufferData, screenWidth, screenHeight,
				translate.x, translate.y, zoom, ppc, cyclicWorld, true);
		}
		else if (d_cpuDisplayData != nullptr) {
			runVisualization(d_cpuDisplayData, worldWidth, worldHeight, d_textureBufferData, screenWidth, screenHeight,
				translate.x, translate.y, zoom, ppc, cyclicWorld, true);
		}
		resetWorldPostprocessDisplay = false;

		checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaPboResource, 0));
	}

	float runCpuLife(size_t iteratinos) {

		size_t worldSize = worldWidth * worldHeight;
		assert(worldWidth % 8 == 0);
		size_t encWorldSize = worldSize / 8;

		if (!cpuLife.areBuffersAllocated(bitLife)) {
			freeAllBuffers();
			if (!cpuLife.allocBuffers(bitLife)) {
				return std::numeric_limits<float>::quiet_NaN();
			}

			initWorld(false, bitLife);

			assert(d_cpuDisplayData == nullptr);
			checkCudaErrors(cudaMalloc((void**)&d_cpuDisplayData, encWorldSize));
		}

		auto t1 = std::chrono::high_resolution_clock::now();
		bool result = cpuLife.iterate(iteratinos);
		auto t2 = std::chrono::high_resolution_clock::now();

		if (!result) {
			return std::numeric_limits<float>::quiet_NaN();
		}

		cpuLife.encodeDataToBpc();

		checkCudaErrors(cudaMemcpy(d_cpuDisplayData, cpuLife.getBpcLifeData(), encWorldSize,
			cudaMemcpyHostToDevice));

		return std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f;
	}

	float runCudaLife(size_t iteratinos, ushort threadsCount, bool bitLife, uint bitLifeBytesPerTrhead) 
	{
		if (!gpuLife.areBuffersAllocated(bitLife)) {
			freeAllBuffers();

			if (!gpuLife.allocBuffers(bitLife)) {
				return std::numeric_limits<float>::quiet_NaN();
			}
			initWorld(true, bitLife);
		}


		auto t1 = std::chrono::high_resolution_clock::now();
		bool result = gpuLife.iterate(iteratinos,threadsCount, bitLife, bitLifeBytesPerTrhead);
		auto t2 = std::chrono::high_resolution_clock::now();

		if (!result) {
			return std::numeric_limits<float>::quiet_NaN();
		}

		return std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f;
	}

	void drawTexture() {
		glColor3f(1.0f, 1.0f, 1.0f);
		glBindTexture(GL_TEXTURE_2D, gl_texturePtr);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pixelBufferObject);

		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, screenWidth, screenHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);


		glBegin(GL_QUADS);
		glTexCoord2f(0.0f, 0.0f);
		glVertex2f(0.0f, 0.0f);
		glTexCoord2f(1.0f, 0.0f);
		glVertex2f(float(screenWidth), 0.0f);
		glTexCoord2f(1.0f, 1.0f);
		glVertex2f(float(screenWidth), float(screenHeight));
		glTexCoord2f(0.0f, 1.0f);
		glVertex2f(0.0f, float(screenHeight));
		glEnd();

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	bool initOpenGlBuffers(int width, int height) {
		// Free any previously allocated buffers

		delete[] h_textureBufferData;
		h_textureBufferData = nullptr;

		glDeleteTextures(1, &gl_texturePtr);
		gl_texturePtr = 0;

		if (gl_pixelBufferObject) {
			cudaGraphicsUnregisterResource(cudaPboResource);
			glDeleteBuffers(1, &gl_pixelBufferObject);
			gl_pixelBufferObject = 0;
		}

		// Check for minimized window or invalid sizes.
		if (width <= 0 || height <= 0) {
			return true;
		}

		// Allocate new buffers.

		h_textureBufferData = new uchar4[width * height];

		glEnable(GL_TEXTURE_2D);
		glGenTextures(1, &gl_texturePtr);
		glBindTexture(GL_TEXTURE_2D, gl_texturePtr);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_textureBufferData);

		glGenBuffers(1, &gl_pixelBufferObject);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pixelBufferObject);
		glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(uchar4), h_textureBufferData, GL_STREAM_COPY);
		cudaError result = cudaGraphicsGLRegisterBuffer(&cudaPboResource, gl_pixelBufferObject, cudaGraphicsMapFlagsWriteDiscard);
		if (result != cudaSuccess) {
			return false;
		}

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		glBindTexture(GL_TEXTURE_2D, 0);
		return true;
	}

	void runLife() {
		float time;
		if (runGpuLife) {
			time = runCudaLife(1, threadsCount, bitLife, bitLifeBytesPerTrhead);
		}
		else {
			time = runCpuLife(1);
		}
		printf("Process Time %f ms\n", time);
	}

	void displayCallback() {

		if (lifeRunning) {
			runLife();
		}

		displayLife();
		drawTexture();
		glutSwapBuffers();
		glutReportErrors();
	}

	void keyboardCallback(unsigned char key, int /*mouseX*/, int /*mouseY*/) {

		switch (key) {
			case 27:  // Esc
				freeAllBuffers();
				exit(EXIT_SUCCESS);
			case ' ':  // Space - run the life iteration step.
				runLife();
				break;
			case '=':
				newWorldWidth += 256;
				printf("Width: %I64u\n", newWorldWidth);
				resizeLifeWorld(newWorldWidth, newWorldHeight);
				resetWorldPostprocessDisplay = true;
				break;
			case '-':
				if (newWorldWidth > 256) {
					newWorldWidth -= 256;
				}
				printf("Width: %I64u\n", newWorldWidth);
				resizeLifeWorld(newWorldWidth, newWorldHeight);
				resetWorldPostprocessDisplay = true;
				break;
			case ']':
				newWorldHeight += 256;
				printf("Height: %I64u\n", newWorldHeight);
				resizeLifeWorld(newWorldWidth, newWorldHeight);
				resetWorldPostprocessDisplay = true;
				break;
			case '[':
				if (newWorldHeight > 256) {
					newWorldHeight -= 256;
				}
				printf("Height: %I64u\n", newWorldHeight);
				resizeLifeWorld(newWorldWidth, newWorldHeight);
				resetWorldPostprocessDisplay = true;
				break;

			case 'g':
				runGpuLife = !runGpuLife;
				if (runGpuLife) {
					printf("GPU\n");
				}
				else {
					printf("CPU\n");
				}
				break;
			case 'b':
				bitLife = !bitLife;
				if (runGpuLife) {
					if (bitLife) {
						printf("Bit per cell\n");
					}
					else {
						printf("Byte per cell\n");
					}
				}
				break;
			case 'n':
				if (bitLifeBytesPerTrhead < 1024) {
					bitLifeBytesPerTrhead <<= 1;
					printf("Byte per thread: %d\n", bitLifeBytesPerTrhead);
				}
				break;
			case 'm':
				if (bitLifeBytesPerTrhead > 1) {
					bitLifeBytesPerTrhead >>= 1;
					printf("Byte per thread: %d\n", bitLifeBytesPerTrhead);
				}
				break;
			default:
				return;
		}

		glutPostRedisplay();
	}

	void mouseCallback(int button, int state, int x, int y) {

		if (button == 3 || button == 4) {
			if (state == GLUT_UP) {
				return; 
			}
			int zoomFactor = (button == 3) ? -1 : 1;
			zoom += zoomFactor;
			resetWorldPostprocessDisplay = true;
			glutPostRedisplay();
			return;
		}
		if (state == GLUT_DOWN) {
			mouseButtons |= 1 << button;
		}
		else if (state == GLUT_UP) {
			mouseButtons &= ~(1 << button);
		}

		mousePosition.x = x;
		mousePosition.y = y;

		glutPostRedisplay();
	}

	void motionCallback(int x, int y) {
		int dx = x - mousePosition.x;
		int dy = y - mousePosition.y;

		if (mouseButtons == 1 << GLUT_LEFT_BUTTON) {
			translate.x += dx;
			translate.y += dy;
			resetWorldPostprocessDisplay = true;
		}

		mousePosition.x = x;
		mousePosition.y = y;

		glutPostRedisplay();
	}

	void reshapeCallback(int w, int h) {
		screenWidth = w;
		screenHeight = h;

		glViewport(0, 0, screenWidth, screenHeight);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0.0, screenWidth, screenHeight, 0.0, -1.0, 1.0);

		initOpenGlBuffers(screenWidth, screenHeight);
		resetWorldPostprocessDisplay = true;
	}

	void idleCallback() {
		if (lifeRunning){
			glutPostRedisplay();
		}
	}

	bool initGL(int* argc, char** argv) {
		glutInit(argc, argv);  // Create GL context.
		glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
		glutInitWindowSize(screenWidth, screenHeight);
		glutCreateWindow("EECS468 Final Project by Yifeng Chen and Junlin Liu");

		glewInit();

		if (!glewIsSupported("GL_VERSION_2_0")) {
			std::cerr << "ERROR: Support for necessary OpenGL extensions missing." << std::endl;
			return false;
		}

		glutReportErrors();
		return true;
	}

	int runGui(int argc, char** argv) {

		if (!initGL(&argc, argv)) {
			return 1;
		}

		printf("-|= World width\n]|[ World height\ng   GPU/CPU: \nb   bit/byte per cell\n");

		// Register GLUT callbacks.
		glutDisplayFunc(displayCallback);
		glutKeyboardFunc(keyboardCallback);
		glutMouseFunc(mouseCallback);
		glutMotionFunc(motionCallback);
		glutReshapeFunc(reshapeCallback);
		//glutIdleFunc(idleCallback);

		assert(sizeof(ubyte) == 1);

		// Init life.
		resizeLifeWorld(newWorldWidth, newWorldHeight);
		runLife();

		// Do not terminate on window close to properly free all buffers.
		glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

		glutMainLoop();

		freeAllBuffers();
		return 0;
	}

}


int main(int argc, char** argv) {
	return mf::runGui(argc, argv);
}

