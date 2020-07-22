#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <time.h>
#include <ctime>
#include <cstdint>
#include <algorithm>
#include <chrono>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/operation.hpp> 
using namespace boost::numeric::ublas;

void conv2d(matrix<int> in, matrix<int> ker, matrix<int> out)
{
    // find center position of kernel (half of kernel size)
    int kCenterX = in.size1() / 2;
    int kCenterY = in.size2() / 2;
    int kRows = ker.size1();
    int kCols = ker.size2();

    for(int i=0; i < in.size1(); ++i)               // rows
    {
        for(int j=0; j < in.size2(); ++j)           // columns
        {
            for(int m=0; m < kRows; ++m)            // kernel rows
            {
                int mm = kRows - 1 - m;             // row index of flipped kernel

                for(int n=0; n < kCols; ++n)        // kernel columns
                {
                    int nn = kCols - 1 - n;         // column index of flipped kernel

                    // index of input signal, used for checking boundary
                    int ii = i + (kCenterY - mm);
                    int jj = j + (kCenterX - nn);

                    // ignore input samples which are out of bound
                    if( ii >= 0 && ii < in.size1() && jj >= 0 && jj < in.size2() )
                        out(i,j) += in(ii,jj) * ker(mm,nn);
                }
            }
        }
    }
}

void MC(std::ofstream& out, int m, int n, int r, double d, int thd){

    // initialize matrix A
    matrix<int> m_a(m,n);
    // if A is a sparse matrix 
    if (d <= 0.5){
        int count_a = m * n * d;
        for (int it = 0; it < count_a; it++){
            // approximation
            int i = rand() % m;
            int j = rand() % n;
            m_a(i,j) = rand() % 1024 + 1;
        }
    }
    // if A is a dense matrix
    else{
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n; j++){
                 m_a(i,j) = rand() % 1024 + 1;
            }
       
        }
    }
    
    // initialize dense matrix B
    matrix<int> m_b(r,r);

    for (int i = 0; i < r; i++){
        for (int j = 0; j < r; j++){
             m_b(i,j) = rand() % 10 + 1;
        }
       
    }
    
    matrix<int> result(m, n);

    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();

    conv2d(m_a, m_b, result);

    auto end_time = high_resolution_clock::now();
    auto time = duration_cast<microseconds>(end_time - start_time).count();

    int c = (m-r+1)*(n-r+1)*r*r;
    // output results
    out << time/1e6 << "," ;
    out << m << "," << n << ",";
    out << r << "," << d << ",";
    out << thd << "," << c << ",\n";

}

int main() {
    // uncomment the following line for reproducibility
    // srand(10);
    
    // customize output filename
    std::string FILENAME = "matrix_conv_cpu_500_points_I5_boost.csv";
    // number of instances of data generated
    int NUM = 500;
    // number of threads are using duing multithreading on CPU
    int N_THD = 1;
    // open the output file
    std::ofstream ofile;
    ofile.open(FILENAME); 
    // initialize each parameter
    for (int i = 0; i < NUM; i++ ){
        int m = rand() % 1024 + 10;
        int n = rand() % 1024 + 10;
        int r = (rand() % 3 + 1) * 2 + 1;
        
        int power;
        double d;

        power = rand()%int((log2(double(m*n))+1));
        d = 1 / pow(2,power);

        int thd = rand() % N_THD + 1;
        // perform kernel operation 
        MC(ofile, m, n, r, d, thd);

        if (i % 10 == 0) std::cout << i << std::endl;
    }

    ofile.close();

    return 0;
}