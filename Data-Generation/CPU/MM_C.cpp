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
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/Core"
using namespace Eigen;
using namespace std;

void doMultiplicationSparseSparse(std::ofstream& out, int m, int n, int k, double d1, double d2, int thd){

    initParallel();
    setNbThreads(thd);
    //cout << "Core: " << nbThreads() << endl;



    // [m*n] * [n*k]

    // initialize sparse matrix A
    SparseMatrix<int> m_a(m, n); 
    int count_a = m * n * d1;
    for (int it = 0; it < count_a; it++){
        int i =  rand() % m;
        int j = rand() % n;

        m_a.coeffRef(i,j) = rand() % 1024 + 1;
    }

    m_a = m_a.pruned();

    // initialize sparse matrix B
    SparseMatrix<int> m_b (n,k);
    int count_b = n * k * d2;
    for (int it = 0; it < count_b; it++){
        int i =  rand() % n;
        int j = rand() % k;

        m_b.coeffRef(i,j) = rand() % 1024 + 1;
    }
    m_b = m_b.pruned();


    Matrix<int,Dynamic,Dynamic> m_c (m, k);


    // do multiplicaition

    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();

    m_c = m_a*m_b;

    auto end_time = high_resolution_clock::now();
    auto time = duration_cast<microseconds>(end_time - start_time).count();

    int c = m*n*k;

    out << time/1e6;
    out << "," << m << "," << n << ",";
    out << k << "," << d1 << "," << d2 << "," << thd << "," << c << ",\n";


}

void doMultiplicationSparseDense(std::ofstream& out, int m, int n, int k, double d1, double d2, int thd){

    initParallel();
    setNbThreads(thd);

    // [m*n] * [n*k]
    // initialize sparse matrix A

    SparseMatrix<int> m_a(m, n); 

    int count_a = m * n * d1;
    for (int it = 0; it < count_a; it++){
        int i =  rand() % m;
        int j = rand() % n;

        m_a.coeffRef(i,j) = rand() % 1024 + 1;
    }

    m_a = m_a.pruned();




    // initialize dense matrix B
    Matrix<int, Dynamic, Dynamic> m_b (n,k);

    int count_b = n * k * d2;
    for (int it = 0; it < count_b; it++){
        int i =  rand() % n;
        int j = rand() % k;

        m_b(i,j) = rand() % 1024 + 1;
    }





    Matrix<int,Dynamic,Dynamic> m_c (m, k);



    // do multiplicaition

    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();

    m_c = m_a*m_b;

    auto end_time = high_resolution_clock::now();

    auto time = duration_cast<microseconds>(end_time - start_time).count();
    
    int c = m*n*k;

    out << time/1e6;
    out << "," << m << "," << n << ",";
    out << k << "," << d1 << "," << d2 << "," << thd << "," << c << ",\n";


}

void doMultiplicationDenseSparse(std::ofstream& out, int m, int n, int k, double d1, double d2, int thd){

    initParallel();
    setNbThreads(thd);

    // [m*n] * [n*k]

    // initialize sparse matrix A

    Matrix<int, Dynamic, Dynamic> m_a(m,n);

    int count_a = m * n * d1;
    for (int it = 0; it < count_a; it++){
        int i =  rand() % m;
        int j = rand() % n;

        m_a(i,j) = rand() % 1024 + 1;
    }



    // initialize dense matrix B
    SparseMatrix<int> m_b (n,k);
    int count_b = n * k * d2;
    for (int it = 0; it < count_b; it++){
        int i =  rand() % n;
        int j = rand() % k;

        m_b.coeffRef(i,j) = rand() % 1024 + 1;
    }

    m_b = m_b.pruned();



    Matrix<int,Dynamic,Dynamic> m_c (m, k);


    // do multiplicaition

    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();

    m_c = m_a*m_b;

    auto end_time = high_resolution_clock::now();

    auto time = duration_cast<microseconds>(end_time - start_time).count();

    int c = m*n*k;

    out << time/1e6;
    out << "," << m << "," << n << ",";
    out << k << "," << d1 << "," << d2 << "," << thd << "," << c << ",\n";


}

void doMultiplicationDenseDense(std::ofstream& out, int m, int n, int k, double d1, double d2, int thd){

    initParallel();
    setNbThreads(thd);

    // [m*n] * [n*k]

    // initialize sparse matrix A

    Matrix<int, Dynamic, Dynamic> m_a(m,n);


    int count_a = m * n * d1;
    for (int it = 0; it < count_a; it++){
        int i =  rand() % m;
        int j = rand() % n;

        m_a(i,j) = rand() % 1024 + 1;
    }
    

    // initialize dense matrix B
    Matrix<int, Dynamic, Dynamic> m_b(n,k);
    int count_b = n * k * d2;
    for (int it = 0; it < count_b; it++){
        int i =  rand() % n;
        int j = rand() % k;

        m_b(i,j) = rand() % 1024 + 1;
    }




    Matrix<int,Dynamic,Dynamic> m_c (m, k);



    // do multiplicaition

    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();

    m_c = m_a*m_b;

    auto end_time = high_resolution_clock::now();

    auto time = duration_cast<microseconds>(end_time - start_time).count();

    int c = m*n*k;

    out << time/1e6;
    out << "," << m << "," << n << ",";
    out << k << "," << d1 << "," << d2 << "," << thd << "," << c << ",\n";


}

int main() {

    // change here to customize output filename
    std::string FILENAME = "matrix_matrix_cpu_5000_points.csv";
    // number of instances of data generated
    int NUM = 5000;
    // number of threads are using duing multithreading on CPU
    int N_THD = 4;

    std::ofstream ofile;
    ofile.open(FILENAME); 

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


        if (d1 <= 0.5 and d2 <= 0.5) doMultiplicationSparseSparse(ofile, m, n, k, d1, d2, thd);
        else if (d1 <= 0.5 and d2 > 0.5) doMultiplicationSparseDense(ofile, m, n, k, d1, d2, thd);
        else if (d1 > 0.5 and d2 <= 0.5) doMultiplicationDenseSparse(ofile, m, n, k, d1, d2, thd);
        else doMultiplicationDenseDense(ofile, m, n, k, d1, d2, thd);


    }

    ofile.close();



    return 0;
}
