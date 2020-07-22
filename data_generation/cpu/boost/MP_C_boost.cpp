#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <time.h>
#include <ctime>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include <math.h>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/operation.hpp> 
using namespace boost::numeric::ublas;
using namespace std;

void maxpool(matrix<int> in, matrix<int> ker, int s){
    
    matrix<int> out (in.size1(), in.size2());
    
    for (int y = 0; y < out.size1(); y+=s) {
        for (int x = 0; x < out.size2(); x+=s) {
            for (int i = 0; i < ker.size1(); ++i) {
                for (int j = 0; j < ker.size2(); ++j) {
                    if ((y * ker.size1() + i) >= in.size1() or (x * ker.size2() + j) >= in.size2()) continue;
                    
                    int value = in(y * ker.size1() + i , x * ker.size2() + j);
                    max(out(y,x), value);
                    
                }
            }
        }
    }
}


void MP(std::ofstream& out, int m, int n, int r, double d, int s, int thd){

    // initialize matrix A
    matrix<int> m_a(m,n);
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
    matrix<int> m_b(r,r);

    for (int i = 0; i < r; i++){
        for (int j = 0; j < r; j++){
             m_b(i,j) = rand() % 10 + 1;
        }
       
    }

    matrix<int> result(m, n);

    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();

    maxpool(m_a, m_b, s);

    auto end_time = high_resolution_clock::now();
    auto time = duration_cast<microseconds>(end_time - start_time).count();

    int c = ceil((double)n/s)*r*r*ceil((double)m/s);
    // output results
    out << time/1e6 << "," ;
    out << m << "," << n << ",";
    out << r << "," << s << "," << d << ",";
    out << thd << "," << c << ",\n";


}


int main() {
    // uncomment the following line for reproducibility
    // srand(10);
    
    // customize output filename
    std::string FILENAME = "max_pooling_cpu_500_points_I5_boost.csv";
    // number of instances of data generated
    int NUM = 500;
    // number of threads are using duing multithreading on CPU
    int N_THD = 1;
    // open the output file
    std::ofstream ofile;
    ofile.open(FILENAME); 
    // initialize each parameter
    for (int i = 0; i < NUM; i++ ){
        int m = rand() % 1024 + 5;
        int n = rand() % 1024 + 5;
        int r = rand() % 4 + 2;
        int s = rand() % 2 + 1;
        
        int power;
        double d;

        power = rand()%int((log2(double(m*n))+1));
        d = 1 / pow(2,power);

        int thd = rand() % N_THD + 1;
        // perform kernel operation 
        MP(ofile, m, n, r, d, s, thd);

        if (i % 10 == 0) std::cout << i << std::endl;
    }

    ofile.close();

    return 0;
}