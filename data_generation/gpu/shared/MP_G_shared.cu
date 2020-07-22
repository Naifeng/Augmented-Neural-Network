#include <iostream>
#include <cstdlib>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <fstream>
#include <time.h>
#include <stdlib.h>


#define TILE_WIDTH 16
#define maskCols 5
#define maskRows 5


#define FH 21
#define FW 21

#define TW 32
#define TH 32


// Max 1024 Threads per Block
#define BH 32
#define BW 32

#define DIV_RUP(x, y)   ((x + y - 1) / y)

#define indexToOffset(x, y, channel, heightOffset, widthOffset, w, h) ((channel * h * w) + (heightOffset + y) * w + widthOffset + x)

#define pixel_x(blockWidth, blockWidthOffset, x) ((blockWidth * blockWidthOffset) + x)
#define pixel_y(blockHeight, blockHeightOffset, y) ((blockHeight * blockHeightOffset) + y)

#define shmem_offset(x_offset, y_offset, x, y, pTW, pw, ph) (((y_offset + y + ph) * pTW + (x_offset + x + pw)))


void fillImage(double* image, int c, int h, int w) {
    for (int i = 0; i < c; i++) {
        for (int j = 0; j < h; j++) {
            for (int k = 0; k < w; k++) {
              image[i * h * w + j * w + k] = i * (j + k);
            }
        }
    }
}

__global__ void cudaMaxPool(double* gOutImage, double* gImage, int c, int h, int w, int fw, int fh, int s) {
    // Tile size
    int tw = blockDim.x;
    int th = blockDim.y;

    // Padded tile size
    int pTW = tw + fw - 1;
    int pTH = th + fh - 1;


    extern __shared__ double shmem[];
    
    // Tile offsets in image. Without Padding
    int tileWidthOffset = tw * blockIdx.x;
    int tileHeightOffset = th * blockIdx.y;
    int channel = blockIdx.z;
    
    
    for(int x = threadIdx.x; x < pTW; x += tw) {
        int copy_x = x - fw/2 + tileWidthOffset;
        for(int y = threadIdx.y; y < pTH; y += tw) {
            int copy_y = y - fh/2 + tileHeightOffset;

            // int shmem_idx = shmem_offset(0, 0, x, y, pTW, 0, 0);
            int index = y * pTW + x;

            if (copy_x < 0 || copy_x >= w || copy_y < 0 || copy_y >= h) {
                shmem[index] = 0;
            } 
            else {
                shmem[index] = gImage[indexToOffset(copy_x, copy_y, channel, 0, 0, w ,h)];
            }
        }
    }
  

    __syncthreads();

    // Pixel this thread is responsible for
    int widthOffset = tileWidthOffset + threadIdx.x;
    int heightOffset = tileHeightOffset + threadIdx.y;

    if (widthOffset < 0 || widthOffset >= w || heightOffset < 0 || heightOffset >= h) {
      return;
    }


    double maxValue = shmem[shmem_offset(threadIdx.x, threadIdx.y, 0, 0, pTW, fw/2, fh/2)];
  
  
    for (int x = -fw/2; x <= fw/2; x+=s) {
        for (int y = -fh/2; y <= fh/2; y+=s) {
          
            double value = shmem[shmem_offset(x, y, threadIdx.x, threadIdx.y, pTW, fw/2, fh/2)];
            
            if (value > maxValue) {
                maxValue = value;
            }
        }
    }
  
    gOutImage[indexToOffset(0, 0, channel, heightOffset, widthOffset, w, h)] = maxValue;

}


void MP(double* out, double* in, int c, int h, int w, int fw, int fh, int s, float & gpu_elapsed_time_ms) {

    long int imageSize = sizeof(double) * c * w * h;
    
    // double* cImage = (double*) malloc(imageSize);
    double* cImage = in;
    double* gImage;

    long int outImageSize = sizeof(double) * c * w * h; 
    
    // double* cOutImage = (double*) malloc(outImageSize);
    double* cOutImage = out;
    double* gOutImage;
    

    cudaMalloc((void**) &gImage, imageSize);
    cudaMalloc((void**) &gOutImage, outImageSize);
    cudaMemset((void*) gOutImage, 0, outImageSize);

    // fillImage(cImage, c, h, w);



    cudaMemcpy (gImage, cImage, imageSize, cudaMemcpyHostToDevice);


    // dim3 simpleGrid(C, DIV_RUP(H, BH), DIV_RUP(W, BW));
    // dim3 simpleBlock(BH, BW);

    // if(clock_gettime(CLOCK_MONOTONIC, &start))
    // { printf("CLOCK ERROR. Exiting.\n"); std::exit(EXIT_FAILURE); }
    // cudaMaxPoolSimple<<<simpleGrid, simpleBlock>>>(gOutImage, gImage, c, h, w, fw, fh);
    // if(clock_gettime(CLOCK_MONOTONIC, &end))
    // { printf("CLOCK ERROR. Exiting.\n"); std::exit(EXIT_FAILURE); }
    // CUDA_CALL(cudaGetLastError());
    // printf("Time cuda code %lf sec\n", TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start));

    int shmem_size = sizeof(double) * (TW + FW - 1) * (TH + FH - 1);
    
    dim3 blockDim(TW, TH);
    dim3 gridDim(DIV_RUP(w, TW), DIV_RUP(h, TH), c);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEventRecord(start, 0);


    cudaMaxPool<<<gridDim, blockDim, shmem_size>>>(gOutImage, gImage, c, h, w, fw, fh, s);
    

    cudaMemcpy(cOutImage, gOutImage, outImageSize, cudaMemcpyDeviceToHost);


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);



    free(cImage);
    free(cOutImage);
    cudaFree(gImage);
    cudaFree(gOutImage);
}



int main(){
    
    // number of instances of data generated
    int NUM = 500;
    std::ofstream ofile;
    // customize output filename
    ofile.open("max_pooling_gpu_500_points_Tesla_2.csv");

    for (int iterator = 0; iterator < NUM; iterator++) {

        if (iterator % 10 == 0) std::cout << "iter: " << iterator << std::endl;

        double *in, *out;
        int m = rand() % 1024 + 5;
        int n = rand() % 1024 + 5;
        int is = n * m;

        int r = rand() % 4 + 2;
        int s = rand() % 2 + 1;

        in = new double[is];
        out = new double[is];

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
        } 
        // if A is a dense matrix
        else {
            for (int i = 0; i < m * n; i++) {
                in[i] = rand() % 1024 + 1;
            }
        }

        float time;
        // perform kernel operation 
        MP(out, in, 1, n, m, r, r, s, time);
        
        int c = ceil((double)n/s)*r*r*ceil((double)m/s);

        ofile << time / 1000;
        ofile << "," << m << "," << n << "," << r << "," << s << "," << d << "," << c << ",\n";

    }

    ofile.close();
    return 0;
}