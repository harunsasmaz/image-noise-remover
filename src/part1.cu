/*	
* noise_remover.cpp
*
* This program removes noise from an image based on Speckle Reducing Anisotropic Diffusion
* Y. Yu, S. Acton, Speckle reducing anisotropic diffusion, 
* IEEE Transactions on Image Processing 11(11)(2002) 1260-1270 <http://people.virginia.edu/~sc5nf/01097762.pdf>
* Original implementation is Modified by Burak BASTEM
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cooperative_groups.h>


#define MATCH(s) (!strcmp(argv[ac], (s)))
#define BLOCK_SIZE 256
#define TILE_DIM 32
// returns the current time
static const double kMicro = 1.0e-6;
double get_time() {
    struct timeval TV;
    struct timezone TZ;
    const int RC = gettimeofday(&TV, &TZ);
    if(RC == -1) {
        printf("ERROR: Bad call to gettimeofday\n");
        return(-1);
    }
    return( ((double)TV.tv_sec) + kMicro * ((double)TV.tv_usec) );
}

namespace cg = cooperative_groups;

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T>
struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

// specialize for double to avoid unaligned memory
// access compile errors
template <>
struct SharedMemory<double> {
  __device__ inline operator double *() {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }

  __device__ inline operator const double *() const {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
};

__global__ void warmup(){}


__global__ void compute1(unsigned char* image, float* diff_coef, float* std_dev, int width, int height,
                        float* north, float* south, float* east, float* west)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index = row * width + col;

    if((row < height - 1) && (col < width - 1) && (row > 0 && col > 0))
    {
        north[index] = image[index - width] - image[index];
        south[index] = image[index + width] - image[index];
        west[index] = image[index - 1] - image[index];
        east[index] = image[index + 1] - image[index];
        float gradient_square = ( north[index] * north[index] 
                                + south[index] * south[index] 
                                + west[index]  * west[index] 
                                + east[index]  * east[index] ) / (image[index] * image[index]);
        float laplacian = (north[index] + south[index] + west[index] + east[index]) / image[index];
        float num = (0.5 * gradient_square) - ((1.0 / 16.0) * (laplacian * laplacian));
        float den = 1 + (.25 * laplacian); 
        float std_dev2 = num / (den * den); 
        den = (std_dev2 - std_dev[0]) / (std_dev[0] * (1 + std_dev[0])); 
        diff_coef[index] = 1.0 / (1.0 + den); 
        if (diff_coef[index] < 0) {
            diff_coef[index] = 0;
        } else if (diff_coef[index] > 1){
            diff_coef[index] = 1;
        }
    }
}

__global__ void compute2(unsigned char* image, float* diff_coef, float* north, float* south,
                            float* east, float* west, float lambda, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index = row * width + col;

    if((col <  width - 1) && (row < height - 1) && (row > 0) && (col > 0)){

        float diff_coef_north = diff_coef[index];	
        float diff_coef_south = diff_coef[index + width];	
        float diff_coef_west = diff_coef[index];	        
        float diff_coef_east = diff_coef[index + 1];					
        float divergence = diff_coef_north * north[index] 
                            + diff_coef_south * south[index] 
                            + diff_coef_west * west[index] 
                            + diff_coef_east * east[index];

        image[index] = image[index] + 0.25 * lambda * divergence;
    }
}

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce1(unsigned char *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;

  T mySum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n) {
    mySum += g_idata[i];

    // ensure we don't read out of bounds -- this is optimized away for powerOf2
    // sized arrays
    if (nIsPow2 || i + blockSize < n) mySum += g_idata[i + blockSize];

    i += gridSize;
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = mySum;
  cg::sync(cta);

  // do reduction in shared mem
  if ((blockSize >= 512) && (tid < 256)) {
    sdata[tid] = mySum = mySum + sdata[tid + 256];
  }

  cg::sync(cta);

  if ((blockSize >= 256) && (tid < 128)) {
    sdata[tid] = mySum = mySum + sdata[tid + 128];
  }

  cg::sync(cta);

  if ((blockSize >= 128) && (tid < 64)) {
    sdata[tid] = mySum = mySum + sdata[tid + 64];
  }

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  if (cta.thread_rank() < 32) {
    // Fetch final intermediate sum from 2nd warp
    if (blockSize >= 64) mySum += sdata[tid + 32];
    // Reduce final warp using shuffle
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      mySum += tile32.shfl_down(mySum, offset);
    }
  }

  // write result for this block to global mem
  if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;
}

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce2(unsigned char *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;

  T mySum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n) {
    mySum += g_idata[i] * g_idata[i];

    // ensure we don't read out of bounds -- this is optimized away for powerOf2
    // sized arrays
    if (nIsPow2 || i + blockSize < n) mySum += g_idata[i + blockSize] * g_idata[i + blockSize];

    i += gridSize;
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = mySum;
  cg::sync(cta);

  // do reduction in shared mem
  if ((blockSize >= 512) && (tid < 256)) {
    sdata[tid] = mySum = mySum + sdata[tid + 256];
  }

  cg::sync(cta);

  if ((blockSize >= 256) && (tid < 128)) {
    sdata[tid] = mySum = mySum + sdata[tid + 128];
  }

  cg::sync(cta);

  if ((blockSize >= 128) && (tid < 64)) {
    sdata[tid] = mySum = mySum + sdata[tid + 64];
  }

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  if (cta.thread_rank() < 32) {
    // Fetch final intermediate sum from 2nd warp
    if (blockSize >= 64) mySum += sdata[tid + 32];
    // Reduce final warp using shuffle
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      mySum += tile32.shfl_down(mySum, offset);
    }
  }

  // write result for this block to global mem
  if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;
}


__global__ void standard_dev(float* sums, float* sums2, float* std_dev, int size, int numBlocks)
{
    float sum = 0, sum2 = 0;
    for(int i=0; i < numBlocks; i++){
        sum += sums[i];
        sum2 += sums2[i];
    }

    float mean = sum / size;
    float variance = (sum2 / size) - mean * mean; // --- 3 floating point arithmetic operations
    std_dev[0] = variance / (mean * mean);
}

extern "C" bool isPow2(unsigned int x) { return ((x & (x - 1)) == 0); }

int main(int argc, char *argv[]) {
    // Part I: allocate and initialize variables
    double time_0, time_1, time_2, time_3, time_4, time_5, time_6, time_7, time_8;	// time variables
    time_0 = get_time();
    const char *filename = "input.pgm";
    const char *outputname = "output.png";	
    int width, height, pixelWidth, n_pixels;
    int n_iter = 50;
    float lambda = 0.5;
    float *north_deriv_dev, *south_deriv_dev, *west_deriv_dev, *east_deriv_dev; // device derivatives
    float *sums, *sums2, *std_dev;	// calculation variables
    float *diff_coef_dev;	// diffusion coefficient
    unsigned char *image_dev;
    time_1 = get_time();	
    
    // Part II: parse command line arguments
    if(argc<2) {
        printf("Usage: %s [-i < filename>] [-iter <n_iter>] [-l <lambda>] [-o <outputfilename>]\n",argv[0]);
        return(-1);
    }
    for(int ac=1;ac<argc;ac++) {
        if(MATCH("-i")) {
            filename = argv[++ac];
        } else if(MATCH("-iter")) {
            n_iter = atoi(argv[++ac]);
        } else if(MATCH("-l")) {
            lambda = atof(argv[++ac]);
        } else if(MATCH("-o")) {
            outputname = argv[++ac];
	} else {
        printf("Usage: %s [-i < filename>] [-iter <n_iter>] [-l <lambda>] [-o <outputfilename>]\n",argv[0]);
        return(-1);
        }
    }

    time_2 = get_time();

    // Part III: read image	
    printf("Reading image...\n");
    unsigned char *image = stbi_load(filename, &width, &height, &pixelWidth, 0);
    if (!image) {
        fprintf(stderr, "Couldn't load image.\n");
        return (-1);
    }
    printf("Image Read. Width : %d, Height : %d, nComp: %d\n",width,height,pixelWidth);
    n_pixels = height * width;
    time_3 = get_time();

    // Part IV: allocate variables
    
    cudaMalloc((void**)&north_deriv_dev, sizeof(float) * n_pixels);
    cudaMalloc((void**)&south_deriv_dev, sizeof(float) * n_pixels);
    cudaMalloc((void**)&west_deriv_dev, sizeof(float) * n_pixels);
    cudaMalloc((void**)&east_deriv_dev, sizeof(float) * n_pixels);
    cudaMalloc((void**)&diff_coef_dev, sizeof(float) * n_pixels);
    cudaMalloc((void**)&image_dev, sizeof(unsigned char) * n_pixels);

    cudaMemcpy(image_dev, image, sizeof(unsigned char) * n_pixels, cudaMemcpyHostToDevice);

    const int reduction_blocks = n_pixels/BLOCK_SIZE + (n_pixels % BLOCK_SIZE == 0 ? 0 : 1);
    const int block_row = height/TILE_DIM + (height % TILE_DIM == 0 ? 0 : 1);
    const int block_col = width/TILE_DIM + (width % TILE_DIM == 0 ? 0 : 1);
    const dim3 blocks(block_col, block_row), threads(TILE_DIM,TILE_DIM);

    cudaMalloc((void**)&sums, sizeof(float)*reduction_blocks);
    cudaMalloc((void**)&sums2, sizeof(float)*reduction_blocks);
    cudaMalloc((void**)&std_dev, sizeof(float));

    int numblocks = reduction_blocks/2 + (reduction_blocks % 2 == 0 ? 0 : 1);
    bool pow2 = isPow2(n_pixels);
    
    // warm up kernel
    warmup<<<blocks, threads>>>();

    time_4 = get_time();
     // Part V: compute --- n_iter * (3 * height * width + 42 * (height-1) * (width-1) + 6) floating point arithmetic operations in totaL
    for (int iter = 0; iter < n_iter; iter++) {

        if(pow2){
            reduce1<float, BLOCK_SIZE, true><<<reduction_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(image_dev, sums, n_pixels);
            reduce2<float, BLOCK_SIZE, true><<<reduction_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(image_dev, sums2, n_pixels);    
        } else {
            reduce1<float, BLOCK_SIZE, false><<<reduction_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(image_dev, sums, n_pixels);
            reduce2<float, BLOCK_SIZE, false><<<reduction_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(image_dev, sums2, n_pixels);    
        }
       
	      standard_dev<<<1,1>>>(sums, sums2, std_dev, n_pixels, numblocks);
	
        compute1<<<blocks, threads>>>(image_dev, diff_coef_dev, std_dev, width, height,
            north_deriv_dev, south_deriv_dev, east_deriv_dev, west_deriv_dev);

        compute2<<<blocks, threads>>>(image_dev, diff_coef_dev, north_deriv_dev, south_deriv_dev,
            east_deriv_dev, west_deriv_dev, lambda, width, height);
	
        cudaDeviceSynchronize();
    }

    time_5 = get_time();

    // Part VI: write image to file
    cudaMemcpy(image, image_dev, sizeof(unsigned char)*n_pixels, cudaMemcpyDeviceToHost);
    stbi_write_png(outputname, width, height, pixelWidth, image, 0);
    time_6 = get_time();

    // Part VII: get average of sum of pixels for testing and calculate GFLOPS
    // FOR VALIDATION - DO NOT PARALLELIZE
    float test = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            test += image[i * width + j];
        }
    }
    test /= n_pixels;	

    float gflops = (float) (n_iter * 1E-9 * (3 * height * width + 42 * (height-1) * (width-1) + 6)) / (time_5 - time_4);
    time_7 = get_time();

    // Part VII: deallocate variables
    stbi_image_free(image);
    cudaFree(north_deriv_dev);
    cudaFree(south_deriv_dev);
    cudaFree(east_deriv_dev);
    cudaFree(west_deriv_dev);
    cudaFree(diff_coef_dev);
    cudaFree(image_dev);
    cudaFree(std_dev);
    cudaFree(sums);
    cudaFree(sums2);
    time_8 = get_time();

    // print
    printf("Time spent in different stages of the application:\n");
    printf("%9.6f s => Part I: allocate and initialize variables\n", (time_1 - time_0));
    printf("%9.6f s => Part II: parse command line arguments\n", (time_2 - time_1));
    printf("%9.6f s => Part III: read image\n", (time_3 - time_2));
    printf("%9.6f s => Part IV: allocate variables\n", (time_4 - time_3));
    printf("%9.6f s => Part V: compute\n", (time_5 - time_4));
    printf("%9.6f s => Part VI: write image to file\n", (time_6 - time_5));
    printf("%9.6f s => Part VII: get average of sum of pixels for testing and calculate GFLOPS\n", (time_7 - time_6));
    printf("%9.6f s => Part VIII: deallocate variables\n", (time_7 - time_6));
    printf("Total time: %9.6f s\n", (time_8 - time_0));
    printf("Average of sum of pixels: %9.6f\n", test);
    printf("GFLOPS: %f\n", gflops);
    return 0;
}
 
 
