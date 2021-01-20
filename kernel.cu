#include <iostream>
#include <opencv2\opencv.hpp>
#include <cuda_runtime.h>

/*
	N = number of pixels
	K = number of superpixels
	S = distance between cluster centers
*/

//namespace
using namespace std;
using namespace cv;

//constant device memory for neigbourhood for contour drawing
__constant__ int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
__constant__ int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

//create data structure named settings
struct Settings {

	//host variables
	float h_W;
	int h_Iterations;
	int h_Spx_Width, h_Spx_Height, h_SpxArea, h_K, h_N, h_Width, h_Height;

	//cluester buffer host
	float* host_fClusters;

	//device variables
	uchar3* device_image;
	float* device_fClusters;
	float* device_fAttributions;
	float* device_labels;

	//cuda arrays
	cudaArray* cuda_Array_BGRA;
	cudaArray* cuda_Array_RGB;
	cudaArray* cuda_Array_Labels;

	//Texture Object
	cudaTextureObject_t tex_Object_BGRA;
	
	//Surface Object
	cudaSurfaceObject_t surf_Object_RGB;
	cudaSurfaceObject_t surf_Object_Labels;
};
struct Settings settings;

//define host functions
void initialize(const Mat& frame);
void segment(const Mat& frame_BGR);
void displayBound(Mat& image);

//kernel for convert uchar to float
__global__ void Cvt_data_type(const cudaTextureObject_t tex_BGRA, cudaSurfaceObject_t surf_RGB, int width, int height) {

	int px = blockIdx.x*blockDim.x + threadIdx.x;
	int py = blockIdx.y*blockDim.y + threadIdx.y;

	uchar4 Pixel = tex2D<uchar4>(tex_BGRA, px, py);
	
	float4 fPixel;

	fPixel.x = Pixel.x;
	fPixel.y = Pixel.y;
	fPixel.z = Pixel.z;
	fPixel.w = Pixel.w;

	surf2Dwrite(fPixel, surf_RGB, px * 16, py);
}

//kernel for initialize cluster centers
__global__ void Initial_cluster_centers(const cudaSurfaceObject_t surf_RGB, float* clusters, int width, int height, int Spx_Per_Row, int Spx_Per_Col) {
	int centroid_Idx = blockIdx.x*blockDim.x + threadIdx.x;
	int K = Spx_Per_Col * Spx_Per_Row;

	if (centroid_Idx < K) {

		int Spx_height = height / Spx_Per_Col;
		int Spx_width = width / Spx_Per_Row;

		int i = centroid_Idx / Spx_Per_Row;
		int j = centroid_Idx % Spx_Per_Row;

		int x = j * Spx_width + Spx_width / 2;
		int y = i * Spx_height + Spx_height / 2;

		float4 color;
		surf2Dread(&color, surf_RGB, x * 16, y);
		clusters[centroid_Idx] = color.x;
		clusters[centroid_Idx + K] = color.y;
		clusters[centroid_Idx + 2 * K] = color.z;
		clusters[centroid_Idx + 3 * K] = x;
		clusters[centroid_Idx + 4 * K] = y;
	}
}

//kernel for assign pixels to clusters 
__global__ void Assignment(const cudaSurfaceObject_t surf_RGB, const float* clusters, const int width, const int height, const int Spx_width, const int Spx_height, const float W, cudaSurfaceObject_t surf_Labels, float* Attribution) {
	
	__shared__ float4 shared_RGB[3][3];
	__shared__ float2 shared_XY[3][3];

	int Clusters_Per_Row = width / Spx_width;
	int K = width / Spx_width * height / Spx_height;
	
	if (threadIdx.x < 3 && threadIdx.y < 3) {
		int id_x = threadIdx.x-1;
		int id_y = threadIdx.y-1;

		int cluster_Idx = blockIdx.x + id_y * Clusters_Per_Row + id_x;

		if (cluster_Idx >= 0 && cluster_Idx < gridDim.x) {
			shared_RGB[threadIdx.y][threadIdx.x].x = clusters[cluster_Idx];
			shared_RGB[threadIdx.y][threadIdx.x].y = clusters[cluster_Idx + K];
			shared_RGB[threadIdx.y][threadIdx.x].z = clusters[cluster_Idx + 2 * K];

			shared_XY[threadIdx.y][threadIdx.x].x = clusters[cluster_Idx + 3 * K];
			shared_XY[threadIdx.y][threadIdx.x].y = clusters[cluster_Idx + 4 * K];
		} 
		else {
			shared_RGB[threadIdx.y][threadIdx.x].x = -1;
		}
	}
	
	__syncthreads();

	//Nearest neighbour
	float S = sqrtf(Spx_width * Spx_height);
	float min_distance = INFINITY;
	float minimum_label = -1;

	int px_in_grid = blockIdx.x*blockDim.x + threadIdx.x;
	int py_in_grid = blockIdx.y*blockDim.y + threadIdx.y;

	int px = px_in_grid % width;

	if (py_in_grid < Spx_height && px < width) {
		int py = py_in_grid + px_in_grid / width * Spx_height;

		float4 color;
		surf2Dread(&color, surf_RGB, px * 16, py);

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {

				if (shared_RGB[i][j].x != -1) {

					float2 px_c_xy = make_float2(px - shared_XY[i][j].x, py - shared_XY[i][j].y);
					float3 px_c_Lab = make_float3(color.x - shared_RGB[i][j].x, color.y - shared_RGB[i][j].y, color.z - shared_RGB[i][j].z);

					//distance calculation
					float ds = sqrt(pow(px_c_xy.x, 2) + pow(px_c_xy.y, 2));
					float dc = sqrt(pow(px_c_Lab.x, 2) + pow(px_c_Lab.y, 2) + pow(px_c_Lab.z, 2));
					float dist = sqrt(dc + ds / S * W);

					float distTmp = fminf(dist, min_distance);

					if (distTmp != min_distance) {
						min_distance = distTmp;
						minimum_label = blockIdx.x + (i-1)*Clusters_Per_Row + (j-1);
					}
				}
			}
		}
		surf2Dwrite(minimum_label, surf_Labels, px * 4, py);
		
		int index_min_label = int(minimum_label);
		atomicAdd(&Attribution[index_min_label], color.x);
		atomicAdd(&Attribution[index_min_label + K], color.y);
		atomicAdd(&Attribution[index_min_label + 2 * K], color.z);
		atomicAdd(&Attribution[index_min_label + 3 * K], px);
		atomicAdd(&Attribution[index_min_label + 4 * K], py);
		atomicAdd(&Attribution[index_min_label + 5 * K], 1);	
	}
}

//kernel for update cluster centers
__global__ void Update(int K, float* clusters, float* Attribution) {

	int cluster_idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (cluster_idx >= K)
		return;

	int counter = Attribution[cluster_idx + K * 5];
		
	if (counter == 0)
		return;

	clusters[cluster_idx]		  = Attribution[cluster_idx] / counter;
	clusters[cluster_idx + K]     = Attribution[cluster_idx + K] / counter;
	clusters[cluster_idx + K * 2] = Attribution[cluster_idx + K * 2] / counter;
	clusters[cluster_idx + K * 3] = Attribution[cluster_idx + K * 3] / counter;
	clusters[cluster_idx + K * 4] = Attribution[cluster_idx + K * 4] / counter;

	Attribution[cluster_idx]         = 0;
	Attribution[cluster_idx + K]     = 0;
	Attribution[cluster_idx + K * 2] = 0;
	Attribution[cluster_idx + K * 3] = 0;
	Attribution[cluster_idx + K * 4] = 0;
	Attribution[cluster_idx + K * 5] = 0;
}

//kernel for draw contours
__global__ void drawContours( const float* labels, uchar3* image, int width, int height) {

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = y * width + x;

	//initialize distance
	int pixel = 0;

	//compare pixel with neighbours
	for (int k = 0; k < 8; k++) {
		int i = dx8[k] + x;
		int j = dy8[k] + y;

		if (i >= 0 && i < width && j >= 0 && j < height) {
			pixel = labels[(idx)] != labels[(j * width + i)] ? pixel + 1 : pixel;
		}
	}

	//change the color if pixel is a contour
	if (pixel >= 2) {
		image[idx].x = 255;
		image[idx].y = 0;
		image[idx].z = 0;
	}
}

int main() {

	VideoCapture cap(0);

	// Parameters
	settings.h_Spx_Height = settings.h_Spx_Width = 20;
	settings.h_W = 35;
	settings.h_Iterations = 1;

	//CUDA functions to get execution time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//start segmentation
	Mat frame;
	cap >> frame;
	initialize(frame);

	while (cap.read(frame)) {
		
		namedWindow("Original", WINDOW_NORMAL);
		imshow("Original", frame);

		cudaEventRecord(start);
		segment(frame);
		cudaEventRecord(stop);
		
		//segmentation time
		float ms = 0.0f;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&ms, start, stop);
		cout << "Process time: " << ms << endl;

		displayBound(frame);
		
		namedWindow("segmentation", WINDOW_NORMAL);
		imshow("segmentation",frame);

		waitKey(1);
	}
	return 0;
}

void initialize(const Mat& frame) {

	//initialize CPU variables
	settings.h_Width = frame.cols;
	settings.h_Height = frame.rows;
	settings.h_N = settings.h_Width * settings.h_Height;
	settings.h_SpxArea = settings.h_Spx_Width * settings.h_Spx_Height;
	settings.h_K = settings.h_N / settings.h_SpxArea;
	settings.host_fClusters = new float[settings.h_K * 5];


	//Initialize GPU buffers
	cudaChannelFormatDesc cuda_Channel_BGRA = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	cudaMallocArray(&settings.cuda_Array_BGRA, &cuda_Channel_BGRA, settings.h_Width, settings.h_Height);

	cudaChannelFormatDesc cuda_Channel_RGB = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	cudaMallocArray(&settings.cuda_Array_RGB, &cuda_Channel_RGB, settings.h_Width, settings.h_Height, cudaArraySurfaceLoadStore);

	cudaChannelFormatDesc cuda_Channel_Labels = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaMallocArray(&settings.cuda_Array_Labels, &cuda_Channel_Labels, settings.h_Width, settings.h_Height, cudaArraySurfaceLoadStore);

	//clusters buffers
	cudaMallocManaged((void**)&settings.device_fClusters, settings.h_K * sizeof(float) * 5);
	cudaMallocManaged((void**)&settings.device_fAttributions, settings.h_K * sizeof(float) * 6);
	cudaMemset(settings.device_fAttributions, 0, settings.h_K * sizeof(float) * 6);
	cudaMallocManaged(&settings.device_image, settings.h_N * 3);
	cudaMallocManaged(&settings.device_labels, settings.h_N * sizeof(float) * 3);

	//object parameters for texture BGRA 
	cudaResourceDesc resource_Desc;
	memset(&resource_Desc, 0, sizeof(resource_Desc));
	resource_Desc.resType = cudaResourceTypeArray;
	resource_Desc.res.array.array = settings.cuda_Array_BGRA;

	cudaTextureDesc texture_Desc;
	memset(&texture_Desc, 0, sizeof(texture_Desc));
	texture_Desc.addressMode[0] = cudaAddressModeClamp;
	texture_Desc.addressMode[1] = cudaAddressModeClamp;
	texture_Desc.filterMode = cudaFilterModePoint;
	texture_Desc.readMode = cudaReadModeElementType;
	texture_Desc.normalizedCoords = false;
	cudaCreateTextureObject(&settings.tex_Object_BGRA, &resource_Desc, &texture_Desc, NULL);

	//surface RGB
	cudaResourceDesc resource_Desc_RGB;
	memset(&resource_Desc_RGB, 0, sizeof(resource_Desc_RGB));
	resource_Desc_RGB.resType = cudaResourceTypeArray;

	resource_Desc_RGB.res.array.array = settings.cuda_Array_RGB;
	cudaCreateSurfaceObject(&settings.surf_Object_RGB, &resource_Desc_RGB);

	//surface labels
	cudaResourceDesc resource_Desc_Labels;
	memset(&resource_Desc_Labels, 0, sizeof(resource_Desc_Labels));
	resource_Desc_Labels.resType = cudaResourceTypeArray;

	resource_Desc_Labels.res.array.array = settings.cuda_Array_Labels;
	cudaCreateSurfaceObject(&settings.surf_Object_Labels, &resource_Desc_Labels);
}

void segment(const Mat& frame_BGR) {

	const int blockW = 16;
	const int blockH = blockW;
	int hMax = 1024 / settings.h_Spx_Height;
	int nBlockPerClust = (settings.h_Spx_Height%hMax == 0) ? settings.h_Spx_Height / hMax : settings.h_Spx_Height / hMax + 1;

	//upload the frame
	Mat frame_BGRA;
	cvtColor(frame_BGR, frame_BGRA, COLOR_BGR2BGRA);
	cudaMemcpyToArray(settings.cuda_Array_BGRA, 0, 0, (uchar*)frame_BGRA.data, settings.h_N * sizeof(uchar4), cudaMemcpyHostToDevice);

	//RGBA2LAB
	dim3 threadsPerBlockColor(blockW, blockH);
	dim3 blocksPerGridColor(settings.h_Width / blockW, settings.h_Height / blockH);

	Cvt_data_type << < blocksPerGridColor, threadsPerBlockColor >> > (settings.tex_Object_BGRA, settings.surf_Object_RGB, settings.h_Width, settings.h_Height);

	cudaDeviceSynchronize();

	//Initialize clusters
	dim3 threadsPerBlockInit(blockW);
	dim3 blocksPerGridInit(settings.h_K / blockW);

	Initial_cluster_centers << < blocksPerGridInit, threadsPerBlockInit >> > (settings.surf_Object_RGB, settings.device_fClusters, settings.h_Width, settings.h_Height, settings.h_Width / settings.h_Spx_Width, settings.h_Height / settings.h_Spx_Height);

	cudaDeviceSynchronize();

	for (int i = 0; i < settings.h_Iterations; i++) {

		//Assignment of supper pixels
		dim3 threadsPerBlocAssignment(settings.h_Spx_Width, min(settings.h_Spx_Height, hMax));
		dim3 blocksPerGridAssignment(settings.h_K, nBlockPerClust);

		Assignment << < blocksPerGridAssignment, threadsPerBlocAssignment >> > (settings.surf_Object_RGB, settings.device_fClusters, settings.h_Width, settings.h_Height, settings.h_Spx_Width, settings.h_Spx_Height, settings.h_W, settings.surf_Object_Labels, settings.device_fAttributions);

		cudaDeviceSynchronize();

		//Update image
		int blocks = (settings.h_K % 1024 == 0) ? settings.h_K % 1024 : settings.h_K % 1024 + 1;
		dim3 threadsPerBlocUpdate(1024);
		dim3 blocksPerGridUpdate(blocks);

		Update << < blocksPerGridUpdate, threadsPerBlocUpdate >> > (settings.h_K, settings.device_fClusters, settings.device_fAttributions);

		cudaDeviceSynchronize();
	}
	cudaMemcpyFromArray(settings.device_labels, settings.cuda_Array_Labels, 0, 0, settings.h_N * sizeof(float), cudaMemcpyDeviceToDevice);
}

//display bounds
void displayBound(Mat& image) {

	dim3 threadsPerBlockDraw(8, 8);
	dim3 blocksPerGridDraw(settings.h_Width / 8, settings.h_Height / 8);

	cudaMemcpy(settings.device_image, image.ptr(), settings.h_N*3, cudaMemcpyHostToDevice);

	drawContours << < blocksPerGridDraw, threadsPerBlockDraw >> > ( settings.device_labels, settings.device_image, settings.h_Width, settings.h_Height);
	
	cudaDeviceSynchronize();

	cudaMemcpy(image.ptr(), settings.device_image, settings.h_N * 3, cudaMemcpyDeviceToHost);
}