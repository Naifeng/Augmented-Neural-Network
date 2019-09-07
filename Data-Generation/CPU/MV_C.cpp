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

void doMultiplication(std::ofstream& out, int m, int n, double d, int thd){



    //cout << omp_get_max_threads() << endl;
    initParallel();
    setNbThreads(thd);
    //cout << "Core: " << nbThreads() << endl;



    // initialize sparse matrix
    SparseMatrix<int> mat (m, n); 
    int count = m * n * d;
    for (int it = 0; it < count; it++){
        int i =  rand() % m;
        int j = rand() % n;
        
        mat.coeffRef(i,j) = rand() % 1024 + 1;
    }


    // initialize vector
    Matrix<int, Dynamic, 1> v (n, 1);


    for (int i = 0; i < n; i++) {
        int value = rand() % 1024;
        v(i) = value;
    }



    Matrix<int,Dynamic,1> k (m, 1);



    // do multiplicaition

    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();

    k = mat.pruned()*v;

    auto end_time = high_resolution_clock::now();

    auto time = duration_cast<microseconds>(end_time - start_time).count();

    int c = m*n;
    out << time/1e6;
    out << "," << m << "," << n << ",";
    out << d << "," << thd << "," << c << ",\n";


}


int main() {

    // change here to customize output filename
    std::string FILENAME = "matrix_vector_cpu_5000_points_I5.csv";
    // number of instances of data generated
    int NUM = 5000;
    // number of threads are using duing multithreading on CPU
    int N_THD = 4;

    std::ofstream ofile;
    ofile.open(FILENAME);

    for (int i = 0; i < NUM; i++ ){
        int m = rand() % 1024 + 1;
        int n = rand() % 1024 + 1;

        int power = rand()%int( log2(double(m*n)) );
        double d;
        if ( power == 0) d = 0.5;
        else d = 1 / pow(2,power);


        int thd = rand() % N_THD + 1; // TODO do not hard code 6

        doMultiplication(ofile, m, n, d, thd);

        if (i % 10 == 0) std::cout << i << std::endl;



    }

    ofile.close();



    return 0;
}
