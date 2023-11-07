//
// Created by albakrih on 04/11/23.
//

#ifndef DEMO_GEMM_H
#define DEMO_GEMM_H

#include "SWTensorBench.h"
#include <immintrin.h>
#include <omp.h>

using namespace swiftware::benchmark;

class GEMMWithPAPI : public SWTensorBench<double>{
    void setup() override {
        Out = new Outputs<double>();
        Out->Out = static_cast<double *>(std::aligned_alloc(32, sizeof(double) * In->Dim1*In->Dim3));
    }

    void preExecute() override {
        for (int i = 0; i < In->Dim1*In->Dim3; ++i) {
            Out->Out[i] = 0;
        }
    }

public:
    GEMMWithPAPI(Inputs<double> *In1, Stats *Stat1) : SWTensorBench<double>(In1, Stat1)
    {}

    ~GEMMWithPAPI(){
    }
};

void GEMM(int m, int n, int k, double *A, double *B, double *C){
    for (int r = 0; r < m; r++) {
        for (int c = 0; c < k ; c++) {
            for (int d = 0 ; d <n ; d++){
                C[r*k+c] += A[r*n+d]*B[d*k+c];
            }
        }
    }
}


//Order : Row, Column, Common dimension
class NaiveGemmRCDWithPapi : public GEMMWithPAPI{
protected:
    Timer execute() override {
        Timer t;
        t.start();

        int m = In->Dim1;
        int n = In->Dim2;
        int k = In->Dim3;
        double *A = In->A;
        double *B = In->x;
        double *C = Out->Out;
        for (int r = 0; r < m; r++) {
            for (int c = 0; c < k ; c++) {
                for (int d = 0 ; d <n ; d++){
                    C[r*k+c] += A[r*n+d]*B[d*k+c];
                }
            }
        }

        t.stop();
        return t;
    }
public:
    NaiveGemmRCDWithPapi(Inputs<double> *In1, Stats *Stat1) : GEMMWithPAPI(In1, Stat1)
    {}
};

class BasicBlockingWithPAPI : public GEMMWithPAPI{
protected:
    Timer execute() override {
        Timer t;
        t.start();

        int M = In->Dim1;
        int N = In->Dim2;
        int K = In->Dim3;
        int bSize = In->Dim4; // x is number of blocks

        double *A = In->A;
        double *B = In->x;
        double *C = Out->Out;
        for (int i = 0 ; i < M; i= i+bSize){
            for (int j = 0; j < K ; j = j+bSize){
                for (int k = 0; k < N; k = k+bSize){
                    for (int ii=i ; ii< i+bSize; ii++){
                        for (int jj = j; jj < j +bSize; jj++) {
                            for (int kk = k ; kk < k+bSize; kk++) {
                                C[ii*K+jj] += A[ii*N+kk]*B[kk*K+jj];
                            }
                        }
                    }
                }
            }
        }
        t.stop();
        return t;
    }
public:
    BasicBlockingWithPAPI(Inputs<double> *In1, Stats *Stat1) : GEMMWithPAPI(In1, Stat1)
    {}
};

class VectorizeBlockingSimple : public GEMMWithPAPI {
protected:
    Timer execute() override {
        Timer t;
        t.start();

        int M = In->Dim1;
        int N = In->Dim2;
        int K = In->Dim3;

        double *A = In->A;
        double *B = In->x;
        double *C = Out->Out;
        for (int i = 0; i < M; i = i + 1) {
            for (int j = 0; j < K; j = j + 1) {

                __m256d c00 = _mm256_set1_pd(0);

                for (int k = 0; k < N; k = k + 8) {

                    __m256d a00 = _mm256_loadu_pd(&A[i * N + k]);
                    __m256d a01 = _mm256_loadu_pd(&A[i * N + k + 4]);

                    __m256d b00 =
                            _mm256_set_pd(B[(k + 3) * K + j], B[(k + 2) * K + j],
                                          B[(k + 1) * K + j], B[k * K + j]);
                    __m256d b01 =
                            _mm256_set_pd(B[(k + 7) * K + j], B[(k + 6) * K + j],
                                          B[(k + 5) * K + j], B[(k + 4) * K + j]);

                    c00 = _mm256_fmadd_pd(b00, a00, c00);
                    c00 = _mm256_fmadd_pd(b01, a01, c00);
                }

                alignas(32) double *temC = new double[4];
                _mm256_storeu_pd(temC, c00);
                C[i * K + j] += temC[0] + temC[1] + temC[2] + temC[3];
            }
        }
        t.stop();
        return t;
    }

public:
    VectorizeBlockingSimple(Inputs<double> *In1, Stats *Stat1)
            : GEMMWithPAPI(In1, Stat1) {}
};

class VecBlockingV2 : public GEMMWithPAPI {
protected:
    Timer execute() override {
        Timer t;
        t.start();

        int M = In->Dim1;
        int N = In->Dim2;
        int K = In->Dim3;
        int bSize = In->Dim4; // x is number of blocks

        double *A = In->A;
        double *B = In->x;
        double *C = Out->Out;
#pragma omp parallel for num_threads(In->NumThreads)
        for (int i = 0; i < M; i = i + bSize) {
            for (int j = 0; j < K; j = j + bSize) {
                for (int k = 0; k < N; k = k + bSize) {
                    for (int ii = i; ii < i + bSize; ii = ii + 4) {
                        for (int jj = j; jj < j + bSize; jj = jj + 8) {
                            __m256d c00 = _mm256_set1_pd(0);
                            __m256d c01 = _mm256_set1_pd(0);
                            __m256d c10 = _mm256_set1_pd(0);
                            __m256d c11 = _mm256_set1_pd(0);
                            __m256d c20 = _mm256_set1_pd(0);
                            __m256d c21 = _mm256_set1_pd(0);
                            __m256d c30 = _mm256_set1_pd(0);
                            __m256d c31 = _mm256_set1_pd(0);

                            __m256d b00;
                            __m256d b01;

                            __m256d a0;
                            __m256d a1;

                            for (int kk = k; kk < k + bSize; kk++) {
                                __builtin_prefetch(&B[kk * K + jj]);
                                __builtin_prefetch(&B[kk * K + jj + 4]);

                                b00 = _mm256_loadu_pd(&B[kk * K + jj]);
                                b01 = _mm256_loadu_pd(&B[kk * K + jj + 4]);

                                a0 = _mm256_set1_pd(A[ii * N + kk]);
                                a1 = _mm256_set1_pd(A[(ii + 1) * N + kk]);

                                c00 = _mm256_fmadd_pd(a0, b00, c00);
                                c01 = _mm256_fmadd_pd(a0, b01, c01);
                                c10 = _mm256_fmadd_pd(a1, b00, c10);
                                c11 = _mm256_fmadd_pd(a1, b01, c11);

                                a0 = _mm256_set1_pd(A[(ii + 2) * N + kk]);
                                a1 = _mm256_set1_pd(A[(ii + 3) * N + kk]);

                                c20 = _mm256_fmadd_pd(a0, b00, c20);
                                c21 = _mm256_fmadd_pd(a0, b01, c21);
                                c30 = _mm256_fmadd_pd(a1, b00, c30);
                                c31 = _mm256_fmadd_pd(a1, b01, c31);
                            }

                            *((__m256d *)(&C[ii * K + jj])) += c00;
                            *((__m256d *)(&C[ii * K + jj + 4])) += c01;
                            *((__m256d *)(&C[(ii + 1) * K + jj])) += c10;
                            *((__m256d *)(&C[(ii + 1) * K + jj + 4])) += c11;
                            *((__m256d *)(&C[(ii + 2) * K + jj])) += c20;
                            *((__m256d *)(&C[(ii + 2) * K + jj + 4])) += c21;
                            *((__m256d *)(&C[(ii + 3) * K + jj])) += c30;
                            *((__m256d *)(&C[(ii + 3) * K + jj + 4])) += c31;
                        }
                    }
                }
            }
        }
        t.stop();
        return t;
    }

public:
    VecBlockingV2(Inputs<double> *In1, Stats *Stat1) : GEMMWithPAPI(In1, Stat1) {}
};


#endif //DEMO_GEMM_H
