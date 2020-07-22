#include <iostream>
#include <cstdlib>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <fstream>
#include <time.h>


__global__ void MaxPool2d(float* bottom_data, const int height, const int width, 
    const int pooled_height,const int out_height,float* top_data, const int st)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int i,j,u,v,index;
    // int index2=x*gridDim.y*out_height*out_height+y*out_height*out_height;
    
    float s;
    for (i = 0; i < out_height; i+=st)
        for (j = 0; j < out_height; j+=st)
        {
            index=x*gridDim.y*height*width+y*height*width+i*pooled_height*width+j*pooled_height;
            s=-10000.0;
            for (u = 0; u < pooled_height&&(u+pooled_height*i)<height; ++u)
                for (v = 0; v < pooled_height&&(v+pooled_height*j)<width; ++v)
                    if (*(bottom_data+index+u*width+v)>s)
                        s=*(bottom_data+index+u*width+v);
            //*(top_data+index2)=s;
            //++index2;
        }
}


void MP(float * input,float* output, int img_height, int img_width, const int r,  const int s, float & gpu_elapsed_time_ms)
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

    MaxPool2d<<<gridsize,blocksize>>>(d_input,img_height,img_width,kernel_height,kernel_width,d_output,s);
    
    cudaMemcpy(output, d_output, img_width*img_height*sizeof(float), cudaMemcpyDeviceToHost);


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
}

int main(){
    // open the output file
    std::ofstream ofile;
    // customize output filename
    ofile.open("max_pooling_gpu_5000_points_Quadro.csv");
    // number of instances of data generated
    int NUM = 5000;


    for (int iterator = 0; iterator < NUM; iterator++) {

        if (iterator % 10 == 0) std::cout << "iter: " << iterator << std::endl;

        float *in, *out;
        int m = rand() % 1024 + 5;
        int n = rand() % 1024 + 5;
        int is = n * m;

        int r = rand() % 4 + 2;
        int s = rand() % 2 + 1;

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
        } 
        // if A is a dense matrix
        else {
            for (int i = 0; i < m * n; i++) {
                in[i] = rand() % 1024 + 1;
            }
        }

        float time;
        // perform kernel operation 
        MP(in, out, n, m, r, s, time);
        int c = ceil((double)n/s)*r*r*ceil((double)m/s);

        ofile << time / 1000;
        ofile << "," << m << "," << n << "," << r << "," << s << "," << d << "," << c << ",\n";


    }

    ofile.close();
    return 0;
}