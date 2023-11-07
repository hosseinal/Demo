#include <iostream>
#include <fstream>
#include "GEMM.h"


int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <NumThreads> <Matrix Size>" << std::endl;
        return 1; // Exit with an error code
    }

    //argv 1 = number of the threads
    //argv 2 is the matrix size
    //argv 3 is header status (1 is print 0 is for not printing)
    //argv 4 is the block size for blocking methods
    int numThreads = 1;
    int matrixSize = 1024;
    int header = 1;
    int numTries = 7;
    int blockSize = 16;
    try {
        numThreads = std::stoi(argv[1]);
        matrixSize = std::stoi(argv[2]);
        header = std::stoi(argv[3]);
        blockSize = std::stoi(argv[4]);
    }catch(const std::exception& e){
        std::cout<<"Number of threads should be an integer";
    }


    auto *in = new Inputs<double>(matrixSize, matrixSize, matrixSize, 0);
    in->ExpName = "ALL_GEMM_With_PAPI";
    in->CorrectSol = static_cast<double *>(std::aligned_alloc(32, sizeof(double) * in->Dim1* in->Dim3));
    // Generate random matrix A
    in->A = static_cast<double *>(std::aligned_alloc(32, sizeof(double) * in->Dim1* in->Dim2));
    for(int i=0 ; i<in->Dim1* in->Dim2; i++){
        in->A[i] = (double) rand() / RAND_MAX;
    }
    //Generate random matrix B
    in->x = static_cast<double *>(std::aligned_alloc(32, sizeof(double) * in->Dim2* in->Dim3));
    for(int i=0 ; i<in->Dim2* in->Dim3; i++) {
        in->x[i] = (double)rand() / RAND_MAX;
    }
    // create matrix c
    in->y = static_cast<double *>(std::aligned_alloc(32, sizeof(double) * in->Dim1*in->Dim3));
    in->NumTrials = numTries;
    in->Threshold = 1e-6;


    //Generate the solution that is correct 100%
    double *CA = static_cast<double *>(std::aligned_alloc(32, sizeof(double) * in->Dim1*in->Dim3));
    GEMM(in->Dim1, in->Dim2,in->Dim3,in->A, in->x,CA);

    for(int i =0 ; i < in->Dim1*in->Dim3 ; i++){
        in->CorrectSol[i] = CA[i];
    }

    swiftware::benchmark::Stats *st;

    auto *pIn = new Inputs<double>(matrixSize, matrixSize, matrixSize, 0);
    pIn->ExpName = "Parallel_GEMM_With_PAPI";
    pIn->CorrectSol = in->CorrectSol;
    pIn->A = in->A;
    pIn->x = in->x;
    pIn->y = in->y;
    pIn->NumThreads = numThreads;
    pIn->NumTrials = numTries;
    pIn->Threshold = 1e-6;

    //Naive GEMM
    st = new Stats("NAIVE_GEMM","MM", pIn->NumTrials);
    auto *naiveGEMM = new NaiveGemmRCDWithPapi(pIn, st);
    naiveGEMM->run();
    auto naiveGemmHeaderStat = naiveGEMM->printStatsHeader();
    auto naiveGemmStat = naiveGEMM->printStats();
    delete naiveGEMM;
    delete st;

    //Basic Blocking
    pIn->Dim4 = blockSize;
    st = new Stats("Basic_Blocking","MM", pIn->NumTrials);
    auto *basicBlocking = new BasicBlockingWithPAPI(pIn, st);
    basicBlocking->run();
    auto blockingHeaderStat = basicBlocking->printStatsHeader();
    auto blockingStat = basicBlocking->printStats();
    delete basicBlocking;
    delete st;

    //Simple Vec GEMM
    st = new Stats("Simple_Vectorize_GEMM","MM", pIn->NumTrials);
    auto *simpleVecGEMM = new VectorizeBlockingSimple(pIn, st);
    simpleVecGEMM->run();
    auto simpleVecGemmHeaderStat = simpleVecGEMM->printStatsHeader();
    auto simpleVecGemmStat = simpleVecGEMM->printStats();
    delete simpleVecGEMM;
    delete st;

    //Vec GEMM V2
    st = new Stats("Vectorize_V2_GEMM","MM", pIn->NumTrials);
    auto *V2VecGEMM = new VecBlockingV2(pIn, st);
    V2VecGEMM->run();
    auto VecGemmV2HeaderStat = V2VecGEMM->printStatsHeader();
    auto VecGemmV2Stat = V2VecGEMM->printStats();
    delete V2VecGEMM;
    delete st;

    auto inHeader = in->printCSVHeader();
    auto inStat = in->printCSV();
    auto pInStat = pIn->printCSV();

    //CSV file for the results
    std::ofstream csvFile("../results.csv", std::ios::app);
    if (!csvFile.is_open()) {
        std::cout << "Error opening the file." << std::endl;
        return 1;
    }

    if (header)
        csvFile<<blockingHeaderStat<<inHeader<<std::endl;
    // Basic Blocking
    csvFile<<blockingStat<<pInStat<<std::endl;
    // NaiveGEMM
    csvFile<<naiveGemmStat<<pInStat<<std::endl;
    // Simple Vec
    csvFile<<simpleVecGemmStat<<pInStat<<std::endl;
    // Vec V2
    csvFile<<VecGemmV2Stat<<pInStat<<std::endl;

    return 0;
}
