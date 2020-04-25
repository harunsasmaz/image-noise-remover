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

#define MATCH(s) (!strcmp(argv[ac], (s)))
#define BLOCK_SIZE 256
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

__global__ void reduction(unsigned char* image, float* sums, float* sums2, int size)
{
    __shared__ float sdata[BLOCK_SIZE];
    __shared__ float sdata2[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float mySum = (i < size) ? image[i] : 0;
    float mySum2 = (i < size) ? image[i] * image[i] : 0;
    if (i + blockDim.x < size){
        mySum += image[i + blockDim.x];
        mySum2 += image[i + blockDim.x] * image[i + blockDim.x];
    } 
    sdata[tid] = mySum;
    sdata2[tid] = mySum2;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
          sdata[tid] = mySum = mySum + sdata[tid + s];
          sdata2[tid] = mySum2 = mySum2 + sdata2[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0){
        sums[blockIdx.x] = mySum;
        sums2[blockIdx.x] = mySum2;
    }
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

    const int reduction_blocks = n_pixels/256 + (n_pixels % 256 == 0 ? 0 : 1);
    const int block_row = height/16 + (height % 16 == 0 ? 0 : 1);
    const int block_col = width/16 + (width % 16 == 0 ? 0 : 1);
    const dim3 blocks(block_col, block_row), threads(16,16);

    cudaMalloc((void**)&sums, sizeof(float)*reduction_blocks);
    cudaMalloc((void**)&sums2, sizeof(float)*reduction_blocks);
    cudaMalloc((void**)&std_dev, sizeof(float));

    // warm up kernel
    // warmup<<<blocks, threads>>>();

    time_4 = get_time();
     // Part V: compute --- n_iter * (3 * height * width + 42 * (height-1) * (width-1) + 6) floating point arithmetic operations in totaL
    for (int iter = 0; iter < n_iter; iter++) {

        reduction<<<reduction_blocks, BLOCK_SIZE>>>(image_dev, sums, sums2, n_pixels);
        //puts(cudaGetErrorString(cudaGetLastError()));
        int numblocks = reduction_blocks/2 + (reduction_blocks % 2 == 0 ? 0 : 1);
	standard_dev<<<1,1>>>(sums, sums2, std_dev, n_pixels, numblocks);
	//puts(cudaGetErrorString(cudaGetLastError()));
	
        compute1<<<blocks, threads>>>(image_dev, diff_coef_dev, std_dev, width, height,
            north_deriv_dev, south_deriv_dev, east_deriv_dev, west_deriv_dev);
	//puts(cudaGetErrorString(cudaGetLastError()));

        compute2<<<blocks, threads>>>(image_dev, diff_coef_dev, north_deriv_dev, south_deriv_dev,
            east_deriv_dev, west_deriv_dev, lambda, width, height);
	
	//puts(cudaGetErrorString(cudaGetLastError()));
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
 
 
