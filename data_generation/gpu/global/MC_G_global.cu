#include <iostream>
#include <cstdlib>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <fstream>
#include <time.h>

__global__ void image_convolution_kernel(float *input, float *out, float *kernelConv,
                                         int img_width, const int img_height,
                                         const int kernel_width, const int kernel_height )
{

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x < img_width) && (y < img_height)){  
        float sum = 0;
        for ( int j = 0; j < kernel_height; j++ )
        {
            for ( int i = 0; i < kernel_width; i++ )
            {
                int dX = x + i - kernel_width / 2;
                int dY = y + j - kernel_height / 2;

                if ( dX < 0 )
                    dX = 0;

                if ( dX >= img_width )
                    dX = img_width - 1;

                if ( dY < 0 )
                    dY = 0;

                if ( dY >= img_height )
                    dY = img_height - 1;


                const int idMat = j * kernel_width + i;
                const int idPixel = dY * img_width + dX;
                sum += (float)input[idPixel] * kernelConv[idMat];
            }
        }

        const int idOut = y * img_width + x;
        out[idOut] = abs(sum);
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

    image_convolution_kernel<<<gridsize,blocksize>>>(d_input,d_output,d_kernel,img_width,img_height,kernel_width,kernel_height);
    cudaMemcpy(output, d_output, img_width*img_height*sizeof(float), cudaMemcpyDeviceToHost);


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
}

int main(){
    // open the output file
    std::ofstream ofile;
    // customize output filename
    ofile.open("matrix_conv_gpu_500_points_Quadro.csv");
    // number of instances of data generated
    int NUM = 500;


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