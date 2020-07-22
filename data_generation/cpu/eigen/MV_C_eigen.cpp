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
using namespace std;

void MV(std::ofstream& out, int m, int n, double d, int thd){
    //cout << omp_get_max_threads() << endl;
    initParallel();
    setNbThreads(thd);
    //cout << "Core: " << nbThreads() << endl;

    // initialize matrix A
    Matrix<int, Dynamic, Dynamic> m_a(m,n);
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
    
    // initialize vector
    Matrix<int, Dynamic, 1> v (n, 1);
    // suppose vector v is a dense vector
    for (int i = 0; i < n; i++) {
        int value = rand() % 1024;
        v(i) = value;
    }

    Matrix<int,Dynamic,1> k (m, 1);

    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();

    k = m_a*v;

    auto end_time = high_resolution_clock::now();
    auto time = duration_cast<microseconds>(end_time - start_time).count();
    // output results
    int c = m*n;
    out << time/1e6;
    out << "," << m << "," << n << ",";
    out << d << "," << thd << "," << c << ",\n";
}


int main() {
    // uncomment the following line for reproducibility
    // srand(10);

    // customize output filename
    std::string FILENAME = "matrix_vector_cpu_5000_points_Xeon.csv";
    // number of instances of data generated
    int NUM = 5000;
    // number of threads are using duing multithreading on CPU
    int N_THD = 64;
    // open the output file
    std::ofstream ofile;
    ofile.open(FILENAME);
    // initialize each parameter
    for (int i = 0; i < NUM; i++ ){
        int m = rand() % 1024 + 1;
        int n = rand() % 1024 + 1;

        int power = rand()%int( log2(double(m*n)) );
        double d;

        d = 1 / pow(2,power);

        int thd = rand() % N_THD + 1; 
        // perform kernel operation 
        MV(ofile, m, n, d, thd);

        if (i % 10 == 0) std::cout << i << std::endl;
    }

    ofile.close();

    return 0;
}