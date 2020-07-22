#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <thrust/device_vector.h>
#include <math.h>

#define BLOCK_SIZE 16
#define TILE_DIM 16

__global__ void MV(int* A, int* B, int* C, int ARows, int ACols, int BRows,
    int BCols, int CRows, int CCols)
{
    int CValue = 0;

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    __shared__ int As[TILE_DIM][TILE_DIM];
    __shared__ int Bs[TILE_DIM][TILE_DIM];

    for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++) {

         if (k*TILE_DIM + threadIdx.x < ACols && Row < ARows)
             As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_DIM + threadIdx.x];
         else
             As[threadIdx.y][threadIdx.x] = 0;

         if (k*TILE_DIM + threadIdx.y < BRows && Col < BCols)
             Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
         else
             Bs[threadIdx.y][threadIdx.x] = 0;

         __syncthreads();

         for (int n = 0; n < TILE_DIM; ++n)
             CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

         __syncthreads();
    }

    if (Row < CRows && Col < CCols)
        C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) +
           (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
}

int main(int argc, char const *argv[])
{
    // number of instances of data generated
    int NUM = 500;
    std::ofstream ofile;
    // change here to customize output filename
    ofile.open("matrix_vector_gpu_500_points_Tesla_2.csv");

    for (int iterator = 0; iterator < NUM; iterator++) { 

        if (iterator % 10 == 0) std::cout << "iter: " << iterator << std::endl;

        // size
        int m, n, k;
        m = rand() % 1024 + 1;
        n = rand() % 1024 + 1;
        k = 1;

        // density
        int power1;
        double d;

        power1 = rand()%int((log2(double(m*n))+1));
        d = 1/pow(2,power1);

        // [m*n] * [n*1]

        // allocate memory in host RAM
        int *h_a, *h_b, *h_c;
        cudaMallocHost((void **) &h_a, sizeof(int) * m * n);
        cudaMallocHost((void **) &h_b, sizeof(int) * n * k);
        cudaMallocHost((void **) &h_c, sizeof(int) * m * k);

        // initialize matrix A
        // if A is a sparse matrix 
        if (d <= 0.5){
            int count_a = m * n * d;
            for (int it = 0; it < count_a; it++){
                // approximation
                int i = rand() % m;
                int j = rand() % n;
                h_a[i*n+j] = rand() % 1024 + 1;
            }
        }
        // if A is a dense matrix
        else{
            for (int i = 0; i < m; i++){
                for (int j = 0; j < n; j++){
                    h_a[i*n+j] = rand() % 1024 + 1;
                }
           
            }
        }

        // random initialize vector B
        int count_b = n;
        for (int it = 0; it < count_b; it++){
            h_b[it] = rand() % 1024 + 1;
        }


        // Allocate memory space on the device
        int *d_a, *d_b, *d_c;
        cudaMalloc((void **) &d_a, sizeof(int) * m * n);
        cudaMalloc((void **) &d_b, sizeof(int) * n * k);
        cudaMalloc((void **) &d_c, sizeof(int) * m * k);

        cudaMemcpy(d_a, h_a, sizeof(int) * m * n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, sizeof(int) * n * k, cudaMemcpyHostToDevice);


        float gpu_elapsed_time_ms;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 dimGrid(grid_cols, grid_rows);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

        cudaEventRecord(start, 0);

        // launch kernel
        MV << < dimGrid, dimBlock >> > (d_a, d_b, d_c, m, n, n, k, m, k);

        cudaMemcpy(h_c, d_c, sizeof(int) * m * k, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);

        int c = m*n;
        ofile << gpu_elapsed_time_ms/1000;
        ofile << "," << m << "," << n << ",";
        ofile << d << "," << c << ",\n";

        // free memory
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaFreeHost(h_a);
        cudaFreeHost(h_b);
        cudaFreeHost(h_c);

    }

    ofile.close();
    return 0;
}