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
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/operation.hpp> 

void MV(std::ofstream& out, int m, int n, double d, int thd){
    using namespace boost::numeric::ublas;
    
    // initialize matrix A
    matrix<int, column_major> m_a(m,n);
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
    vector<int> v(n);
    // suppose vector v is a dense vector
    for (int i = 0; i < n; i++) {
        int value = rand() % 1024;
        v(i) = value;
    }

    vector<int> k (m);

    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();

    boost::numeric::ublas::axpy_prod(m_a,v,k,true);

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
    std::string FILENAME = "matrix_vector_cpu_500_points_Xeon_boost.csv";
    // number of instances of data generated
    int NUM = 500;
    // number of threads are using duing multithreading on CPU
    int N_THD = 1;
    // open the output file
    std::ofstream ofile;
    ofile.open(FILENAME);
    // initialize each parameter
    for (int i = 0; i < NUM; i++ ){
        int m = rand() % 1024 + 1;
        int n = rand() % 1024 + 1;

        int power = rand()%int(log2(double(m*n)));
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