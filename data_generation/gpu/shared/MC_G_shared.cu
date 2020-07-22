#include <iostream>
#include <cstdlib>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <fstream>
#include <time.h>

#define TILE_WIDTH 16
#define maskCols 5
#define maskRows 5
#define w (TILE_WIDTH + maskCols -1)

__global__ void tilingKernelProcessing(float * InputImageData, const float *__restrict__ kernel, 
                                       float* outputImageData, int channels, int width, int height)
{

    __shared__ float N_ds[w][w];  //block of image in shared memory


    // allocation in shared memory of image blocks
    int maskRadius = maskRows/2;
    for (int k = 0; k <channels; k++) {
        int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
        int destY = dest/w;     //row of shared memory
        int destX = dest%w;     //col of shared memory
        int srcY = blockIdx.y *TILE_WIDTH + destY - maskRadius; // index to fetch data from input image
        int srcX = blockIdx.x *TILE_WIDTH + destX - maskRadius; // index to fetch data from input image
        int src = (srcY *width +srcX) * channels + k;   // index of input image
        if(srcY>= 0 && srcY < height && srcX>=0 && srcX < width)
            N_ds[destY][destX] = InputImageData[src];  // copy element of image in shared memory
        else
            N_ds[destY][destX] = 0;



        dest = threadIdx.y * TILE_WIDTH+ threadIdx.x + TILE_WIDTH * TILE_WIDTH;
        destY = dest/w;
        destX = dest%w;
        srcY = blockIdx.y *TILE_WIDTH + destY - maskRadius;
        srcX = blockIdx.x *TILE_WIDTH + destX - maskRadius;
        src = (srcY *width +srcX) * channels + k;
        if(destY < w){
            if(srcY>= 0 && srcY < height && srcX>=0 && srcX < width)
                N_ds[destY][destX] = InputImageData[src];
            else
                N_ds[destY][destX] = 0;
        }

        __syncthreads();


        //compute kernel convolution
        float accum = 0;
        int y, x;
        for (y= 0; y < maskCols; y++)
            for(x = 0; x<maskRows; x++)
                accum += N_ds[threadIdx.y + y][threadIdx.x + x] *kernel[y * maskCols + x];

        y = blockIdx.y * TILE_WIDTH + threadIdx.y;
        x = blockIdx.x * TILE_WIDTH + threadIdx.x;
        if(y < height && x < width)
            outputImageData[(y * width + x) * channels + k] = accum;
        __syncthreads();


    }

}

void MC(float * input,float* output, int img_height, int img_width, const int r, float & gpu_elapsed_time_ms)
{


    // initialize kernel here
    int kernel_height = r;
    int kernel_width = r;

    float *kernel;
    kernel = new float[r*r];

    for (int i = 0; i < r*r; i++){
        kernel[i] = rand() % 10 + 1;
    }


    float * mask = new float[kernel_height*kernel_width];
    for (int i = 0; i < kernel_height*kernel_width; i++)
    {
        mask[i] = kernel[i];
    }

    float * d_input, * d_output, * d_kernel;
    cudaMalloc(&d_input, img_width*img_height*sizeof(float));
    cudaMalloc(&d_output, img_width*img_height*sizeof(float));
    cudaMalloc(&d_kernel, kernel_height*kernel_width*sizeof(float));

    cudaMemcpy(d_input, input, img_width*img_height*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, mask, kernel_height*kernel_width*sizeof(float), cudaMemcpyHostToDevice);
    dim3 blocksize(16,16);
    dim3 gridsize;
    gridsize.x=(img_width+blocksize.x-1)/blocksize.x;
    gridsize.y=(img_height+blocksize.y-1)/blocksize.y;




    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEventRecord(start, 0);
    
    tilingKernelProcessing<<<gridsize,blocksize>>>(d_input, d_kernel, d_output, 1, img_width, img_height);

    cudaMemcpy(output, d_output, img_width*img_height*sizeof(float), cudaMemcpyDeviceToHost);


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
}

int main(){
    
    // number of instances of data generated
    int NUM = 500;

    std::ofstream ofile;
    // customize output filename
    ofile.open("matrix_conv_gpu_500_points_Tesla_2.csv");

    for (int iterator = 0; iterator < NUM; iterator++) {

        if (iterator % 10 == 0) std::cout << "iter: " << iterator << std::endl;

        float *in, *out;
        int m = rand() % 1024 + 10;
        int n = rand() % 1024 + 10;
        int is = n * m;

        int r = (rand() % 3 + 1) * 2 + 1;

        in = new float[is];
        out = new float[is];

        // density
        int power;
        double d;

        power = rand() % int((log2(double(m * n)) + 1));
        d = 1 / pow(2, power);

        // initialize matrix A
        // if A is a sparse matrix 
        if (d <= 0.5) {
            int count_a = m * n * d;
            for (int it = 0; it < count_a; it++) {
                int i = rand() % m;
                int j = rand() % n;

                in[i * n + j] = rand() % 1024 + 1;
            }
        // if A is a dense matrix    
        } else {
            for (int i = 0; i < m * n; i++) {
                in[i] = rand() % 1024 + 1;
            }
        }

        float time;
        // perform kernel operation 
        MC(in, out, n, m, r, time);

        int c = (m-r+1)*(n-r+1)*r*r;
        ofile << time / 1000;
        ofile << "," << m << "," << n << "," << r << "," << d << "," << c << ",\n";

    }

    ofile.close();
    return 0;
}