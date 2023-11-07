//
// Created by albakrih on 06/11/23.
//

#ifndef DEMO_SPMM_H
#define DEMO_SPMM_H

#include "SWTensorBench.h"
#include "aggregation/sparse_io.h"
#include "aggregation/sparse_utilities.h"
#include <immintrin.h>

using namespace swiftware::benchmark;
using namespace std;
template <typename T> struct SparseInput : Inputs<T> {

public:
    sym_lib::CSR *A;
    T *B;
    SparseInput(int Dim1, int Dim2, int Dim3, int Dim4)
            : Inputs<T>(Dim1, Dim2, Dim3, Dim4) {}

    ~SparseInput() { delete A; }
};

sym_lib::CSR *randomSquareMatrix(int size) {
    // Generate random matrix a
    double **a = new double *[size];
    for (int i = 0; i < size; i++) {
        a[i] = new double[size];
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if ((double)rand() / RAND_MAX < 0.2) {
                a[i][j] = (double)rand() / RAND_MAX;
            } else {
                a[i][j] = 0;
            }
        }
    }
    auto *cscA = sym_lib::dense_to_csc(size, size, a);
    return sym_lib::csc_to_csr(cscA);
}

void BaseLine(int dim1, int dim2, int dim3, sym_lib::CSR *A, double *B,
              double *C) {
    // M*N N*K = M*K
    int M = dim1;
    int N = dim2;
    int K = dim3;
    for (int i = 0; i < M; i++) {
        for (int j = A->p[i]; j < A->p[i + 1]; j++) {
            int indexOfB = A->i[j] * K;
            for (int k = 0; k < K; k++) {
                C[i * K + k] += A->x[j] * B[indexOfB + k];
            }
        }
    }
}

class SPMMWithPAPI : public SWTensorBench<double> {
    void setup() override {
        Out = new Outputs<double>();
        Out->Out = static_cast<double *>(
                std::aligned_alloc(32, sizeof(double) * In->Dim1 * In->Dim3));
    }

    void preExecute() override {
        for (int i = 0; i < In->Dim1 * In->Dim3; ++i) {
            Out->Out[i] = 0;
        }
    }

    bool verify(double &Error) override {
        bool retValue = true;
        if (In->CorrectSol == nullptr)
            return true;
        double infNorm = 0;
        for (int i = 0; i < In->Dim1; ++i) {
            if (std::abs(Out->Out[i] - In->CorrectSol[i]) > infNorm) {
                infNorm = std::abs(Out->Out[i] - In->CorrectSol[i]);
            }
        }
        Error = (double)infNorm;
        if (infNorm > In->Threshold) {
            retValue = false;
        }
        return retValue;
    }

public:
    SparseInput<double> *In;
    SPMMWithPAPI(SparseInput<double> *In1, Stats *Stat1)
            : SWTensorBench<double>(In1, Stat1) {
        In = In1;
    }

    ~SPMMWithPAPI() {
        free(Out->Out);
        delete Out;
    }
};

class SpMM : public SPMMWithPAPI {
protected:
    Timer execute() override {
        Timer t;
        t.start();

        // M*N N*K = M*K
        int M = In->Dim1;
        int N = In->Dim2;
        int K = In->Dim3;
        for (int i = 0; i < M; i++) {
            for (int j = In->A->p[i]; j < In->A->p[i + 1]; j++) {
                int indexOfB = In->A->i[j] * K;
                for (int k = 0; k < K; k++) {
                    Out->Out[i * K + k] += In->A->x[j] * In->B[indexOfB + k];
                }
            }
        }

        t.stop();
        return t;
    }

public:
    SpMM(SparseInput<double> *In1, Stats *Stat1) : SPMMWithPAPI(In1, Stat1) {}
};

class SpMMVec : public SPMMWithPAPI {
protected:
    Timer execute() override {
        Timer t;
        t.start();

        // M*N N*K = M*K
        int M = In->Dim1;
        int N = In->Dim2;
        int K = In->Dim3;
        for (int i = 0; i < M; i++) {
            for (int j = In->A->p[i]; j < In->A->p[i + 1]; j++) {
                __m256d a0 = _mm256_set1_pd(In->A->x[j]);
                int indexOfB = In->A->i[j] * K;
                for (int k = 0; k < K; k = k + 4) {
                    __m256d c0 = _mm256_loadu_pd(&Out->Out[i * K + k]);
                    __m256d b0 = _mm256_loadu_pd(&In->B[indexOfB + k]);
                    c0 = _mm256_fmadd_pd(a0,b0,c0);
                    _mm256_storeu_pd(&Out->Out[i * K + k], c0);
                }
            }
        }

        t.stop();
        return t;
    }

public:
    SpMMVec(SparseInput<double> *In1, Stats *Stat1) : SPMMWithPAPI(In1, Stat1) {}
};
#endif //DEMO_SPMM_H
