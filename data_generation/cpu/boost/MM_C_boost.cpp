#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <time.h>
#include <ctime>
#include <cstdint>
#include <algorithm>
#include <math.h>
#include <chrono>
using namespace std;

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/operation.hpp> 

void MM(std::ofstream& out, int m, int n, int k, double d1, double d2, int thd){
    using namespace boost::numeric::ublas;
    
    // [m*n] * [n*k]
    // initialize matrix A
    matrix<int, column_major> m_a(m,n);
    // if A is a sparse matrix 
    if (d1 <= 0.5){
        int count_a = m * n * d1;
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
    
    // initialize matrix B
    matrix<int, column_major> m_b(n,k);
    // if B is a sparse matrix 
    if (d2 <= 0.5){
        int count_b = n * k * d2;
        for (int it = 0; it < count_b; it++){
            // approximation
            int i = rand() % n;
            int j = rand() % k;
            m_b(i,j) = rand() % 1024 + 1;
        }
    }
    // if B is a dense matrix
    else{
        for (int i = 0; i < n; i++){
            for (int j = 0; j < k; j++){
                 m_b(i,j) = rand() % 1024 + 1;
            }
       
        }
    }

    matrix<int, column_major> m_c(m,k);

    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();

    boost::numeric::ublas::axpy_prod(m_a,m_b,m_c,true);

    auto end_time = high_resolution_clock::now();
    auto time = duration_cast<microseconds>(end_time - start_time).count();

    int c = m*n*k;
    // output results
    out << time/1e6;
    out << "," << m << "," << n << ",";
    out << k << "," << d1 << "," << d2 << "," << thd << "," << c << ",\n";

}

int main() {
    // uncomment the following line for reproducibility
    // srand(10);

    // customize output filename
    std::string FILENAME = "matrix_matrix_cpu_500_points_I5_boost.csv";
    // number of instances of data generated
    int NUM = 500;
    // number of threads are using duing multithreading on CPU
    int N_THD = 1;
    // open the output file
    std::ofstream ofile;
    ofile.open(FILENAME); 
    // initialize each parameter
    for (int i = 0; i < NUM; i++){ 
        if (i % 10 == 0) std::cout << i << std::endl;
        int m,n,k;
        m = rand() % 1024 + 1;
        n = rand() % 1024 + 1;
        k = rand() % 1024 + 1;

        int power1, power2;
        double d1 ,d2;

        power1 = rand()%int((log2(double(m*n))+1));
        d1 = 1 / pow(2,power1);

        power2 = rand()%int((log2(double(n*k))+1));
        d2 = 1 / pow(2,power2);

        int thd = rand() % N_THD + 1; 
        // perform kernel operation 
        MM(ofile, m, n, k, d1, d2, thd);

    }

    ofile.close();

    return 0;
}