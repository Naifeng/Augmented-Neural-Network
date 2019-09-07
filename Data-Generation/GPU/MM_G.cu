#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <thrust/device_vector.h>
#include <math.h>


#define BLOCK_SIZE 16

__global__ void gpu_matrix_mult(int *a,int *b, int *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < k && row < m) {
        for(int i = 0; i < n; i++) {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 


int main(int argc, char const *argv[])
{
    
    // change here to customize output filename
    string FILENAME = "matrix_matrix_gpu_5000_points.csv";
    // number of instances of data generated
    int NUM = 5000;

    std::ofstream ofile;
    ofile.open(FILENAME); 


    for (int iterator = 0; iterator < NUM; iterator++) { 

        if (iterator % 10 == 0) std::cout << "iter: " << iterator << std::endl;

        // size
        int m, n, k;
        m = rand() % 1024 + 1;
        n = rand() % 1024 + 1;
        k = rand() % 1024 + 1;


        // density
        int power1, power2;
        double d1,d2;


        power1 = rand()%int((log2(double(m*n))+1));
        d1 = 1/pow(2,power1);

        power2 = rand()%int((log2(double(n*k))+1));
        d2 = 1/pow(2,power2);


        // [m*n] * [n*k]


        // allocate memory in host RAM
        int *h_a, *h_b, *h_c;
        cudaMallocHost((void **) &h_a, sizeof(int) * m * n);
        cudaMallocHost((void **) &h_b, sizeof(int) * n * k);
        cudaMallocHost((void **) &h_c, sizeof(int) * m * k);


        // random initialize matrix A

        int count_a = m * n * d1;
        for (int it = 0; it < count_a; it++){
            int i =  rand() % m;
            int j = rand() % n;

            h_a[i*n+j] = rand() % 1024 + 1;
        }


        // random initialize matrix B

        int count_b = n * k * d2;
        for (int it = 0; it < count_b; it++){
            int i =  rand() % n;
            int j = rand() % k;

            h_b[i*k+j] = rand() % 1024 + 1;
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
        gpu_matrix_mult << < dimGrid, dimBlock >> > (d_a, d_b, d_c, m, n, k);

        cudaMemcpy(h_c, d_c, sizeof(int) * m * k, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);


        cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);

        int c = m*n*k;
        ofile << gpu_elapsed_time_ms/1000;
        ofile << "," << m << "," << n << "," << k << ",";
        ofile << d1 << "," << d2 << "," << c << ",\n";


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