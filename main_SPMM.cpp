//
// Created by albakrih on 07/11/23.
//
#include "SPMM.h"
#include "iostream"
#include "string"

int main(int argc, char *argv[]){
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <NumThreads> <Matrix Size>" << std::endl;
        return 1; // Exit with an error code
    }

    //argv 1 = number of the threads (Note : We didn't use it in this example, but it is available for further implementations
    //argv 2 is the matrix size
    //argv 3 is header status (1 is print 0 is for not printing)
    //argv 4 is the block size for blocking methods
    int numThreads = 1;
    int matrixSize = 1024;
    int header = 1;
    string path = "";
    try {
        numThreads = std::stoi(argv[1]);
        matrixSize = std::stoi(argv[2]);
        header = std::stoi(argv[3]);
    }catch(const std::exception& e){
        std::cout<<"Number of threads should be an integer";
    }
    sym_lib::CSR *A;
    A = randomSquareMatrix(matrixSize);

    auto *in = new SparseInput<double>(A->m, A->n, matrixSize, 0);
    in->A = A;

    // Generate random matrix B
    in->B = static_cast<double *>(std::aligned_alloc(32, sizeof(double) * in->Dim2* in->Dim3));
    for(int i=0 ; i<in->Dim2* in->Dim3; i++){
        in->B[i] = (double) rand() / RAND_MAX;
    }

    //set correct value
    double *CA = static_cast<double *>(std::aligned_alloc(32, sizeof(double) * in->Dim1*in->Dim3));
    in->CorrectSol = static_cast<double *>(std::aligned_alloc(32, sizeof(double) * in->Dim1* in->Dim3));
    BaseLine(in->Dim1,in->Dim2,in->Dim3,in->A,in->B,CA);
    for(int i =0 ; i < in->Dim1*in->Dim3 ; i++){
        in->CorrectSol[i] = CA[i];
    }

    in->ExpName = "ALL_SpMM_With_PAPI";
    in->NumTrials = 7; // set it to more than 3 to get a reasonable measurement

    swiftware::benchmark::Stats *st;
    st = new Stats("Basic_SpMM","MM", in->NumTrials);
    auto *basicSpMM = new SpMM(in, st);
    basicSpMM->run();
    auto spmmHeaderStat = basicSpMM->printStatsHeader();
    auto spmmStat = basicSpMM->printStats();
    delete basicSpMM;
    delete st;

    st = new Stats("Vec_SpMM","MM", in->NumTrials);
    auto *VrcSpMM = new SpMMVec(in, st);
    VrcSpMM->run();
    auto vecHeaderStat = VrcSpMM->printStatsHeader();
    auto vecSpmmStat = VrcSpMM->printStats();
    delete VrcSpMM;
    delete st;

    auto inHeader = in->printCSVHeader();
    auto inStat = in->printCSV();

    std::ofstream csvFile("../results_spmm.csv", std::ios::app);
    if (!csvFile.is_open()) {
        std::cout << "Error opening the file." << std::endl;
        return 1;
    }

    if (header)
        csvFile<<spmmHeaderStat<<inHeader<<std::endl;
    // Basic SPMM
    csvFile<<spmmStat<<in->printCSV()<<std::endl;
    csvFile<<vecSpmmStat<<in->printCSV()<<std::endl;


    free(in->CorrectSol);
    free(CA);
    free(in->B);
    delete in;

    return 0;
}