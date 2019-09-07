#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <thrust/device_vector.h>

#define BLOCK_SIZE 16



__global__ void kernel(int *vec, int *mat, int *out, const int N, const int M){
    int tid=threadIdx.x+blockIdx.x*blockDim.x;
    int sum=0;
    if(tid<M){
        for(int i=0; i<N; i++)
            sum += vec[i]*mat[(i*M)+tid];
        out[tid]=sum;
    }
}

__global__ void matrixvectorMultiplication(int *A, int *B, int *C, int M, int N)
{
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    double value = 0;
    for(int k = 0; k < N; k++){
        value += A[k * M + j] * B[i * M + k];
    }
    C[i * N + j] = value;
}



int main(int argc, char const *argv[])
{
    
    // number of instances of data generated
    int NUM = 5000;

    std::ofstream ofile;

    // change here to customize output filename
    ofile.open("matrix_vector_gpu_5000_points_Tesla.csv");

    // [1*n] * [n*m]
    for (int iterator = 0; iterator <= NUM; iterator++) {

        if (iterator % 10 == 0) std::cout << "iter: " << iterator << std::endl;

        int m, n;
        double d;

        n = rand() % 1024 + 1;
        m = rand() % 1024 + 1;

        int power = rand()%int((log2(double(m*n))+1)); // TODO ONLY CONSIDER SPARSE
        d = 1/pow(2,power);



        thrust::device_vector<int> den(m * n, 0);


        int *h_a, *h_b, *h_c;
        cudaMallocHost((void **) &h_a, sizeof(int) * n);
        cudaMallocHost((void **) &h_b, sizeof(int) * n * m);
        cudaMallocHost((void **) &h_c, sizeof(int) * m);


        // random initialize vector A
        for (int i = 0; i < n; i++)
            h_a[i] = rand() % 1024 + 1;


        // random initialize matrix B

        int count = m * n * d;
        for (int it = 0; it < count; it++){
            int i =  rand() % m;
            int j = rand() % n;

            h_b[i*n+j] = rand() % 1024 + 1;
        }

        // Allocate memory space on the device
        int *d_a, *d_b, *d_c;
        cudaMalloc((void **) &d_a, sizeof(int) * n);
        cudaMalloc((void **) &d_b, sizeof(int) * n * m);
        cudaMalloc((void **) &d_c, sizeof(int) * m);

        // copy matrix A and B from host to device memory
        cudaMemcpy(d_a, h_a, sizeof(int) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, sizeof(int) * n * m, cudaMemcpyHostToDevice);


        float gpu_elapsed_time_ms;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);

        // launch kernel
        kernel <<< m / 256 + 1, 256 >>> (d_a, d_b, d_c, n, m);


        cudaMemcpy(h_c, d_c, sizeof(int) * m, cudaMemcpyDeviceToHost);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);

        int c = m*n;
        ofile << gpu_elapsed_time_ms / 1000;
        ofile << "," << m << "," << n << "," << d << "," << c << ",\n";

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