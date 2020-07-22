#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <time.h>
#include <ctime>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include "Eigen/Dense"
#include "Eigen/Sparse"
using namespace Eigen;

template <typename Derived, typename Derived2 >
Derived conv2d(const MatrixBase<Derived>& I, const MatrixBase<Derived2> &kernel )
{
    Derived O = Derived::Zero(I.rows(),I.cols());
    
    
    typedef typename Derived::Scalar Scalar;
    typedef typename Derived2::Scalar Scalar2;
    
    int col=0,row=0;
    int KSizeX = kernel.rows();
    int KSizeY = kernel.cols();
    
    int limitRow = I.rows()-KSizeX;
    int limitCol = I.cols()-KSizeY;
    
    Derived2 block ;
    Scalar normalization = kernel.sum();
    if ( normalization < 1E-6 )
    {
        normalization=1;
    } 
    for ( row = 0; row < limitRow; row++ )
    {
      
      for ( col = 0; col < limitCol; col++ )
      {    
      Scalar b=(static_cast<Derived2>( I.block(row,col,KSizeX,KSizeY ) ).cwiseProduct(kernel)).sum();
      O.coeffRef(row+KSizeX/2,col+KSizeY/2) = b;
      }
    }
    
    return O/normalization;
}


void MC(std::ofstream& out, int m, int n, int r, double d, int thd){
    //cout << omp_get_max_threads() << endl;
    initParallel();
    setNbThreads(thd);
    //cout << "Core: " << nbThreads() << endl;

    // initialize matrix A
    Matrix<int, Dynamic, Dynamic> m_a(m,n);
    m_a.setZero(m,n);
    // if A is a sparse matrix 
    if (d <= 0.5){
        int count_a = m * n * d;
        for (int it = 0; it < count_a; it++){
            int i =  rand() % m;
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
    Matrix<int, Dynamic, Dynamic> m_b(r,r);

    for (int i = 0; i < r; i++){
        for (int j = 0; j < r; j++){
             m_b(i,j) = rand() % 10 + 1;
        }
       
    }
    
    Matrix<int,Dynamic,Dynamic> result(m, n);
    result.setZero(m,n);

    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();

    result = conv2d(m_a, m_b);

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
    std::string FILENAME = "matrix_conv_cpu_5000_points_I5.csv";
    // number of instances of data generated
    int NUM = 5000;
    // number of threads are using duing multithreading on CPU
    int N_THD = 4;
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

        MC(ofile, m, n, r, d, thd);

        if (i % 10 == 0) std::cout << i << std::endl;
    }

    ofile.close();

    return 0;
}