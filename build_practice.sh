#!/bin/bash
#SBATCH --cpus-per-task=40
#SBATCH --export=ALL
#SBATCH --job-name="swbench"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes

#SBATCH --nodes=1
#SBATCH --output="gemm_log_%j.out"
#SBATCH -t 00:45:00
#SBATCH --constraint=cascade



### This scripts builds the swiftware benchmark on Niagara server (and possibly other linux machines).
### It first installs PAPI and then builds the repository
THREADS=20

module load NiaEnv/.2022a
module load intel/2022u2
module load cmake
module load gcc

echo "----- Building swbench -----"
rm -rf build
# Create build folder
mkdir -p build && cd build

# Configure
# use below line if you want PAPI to be on
#cmake -DPROFILING_WITH_PAPI=ON -DCMAKE_BUILD_TYPE=Release -DPAPI_PREFIX=${HOME}/programs/papi/ ..

#-----------Build on my own device
#cmake -DCMAKE_BUILD_TYPE=Release -DPAPI_PREFIX=${HOME}/programs/papi/ -DMKL_DIR="/home/albakrih/intel/oneapi/mkl/2023.2.0" -DCMAKE_PREFIX_PATH="/home/albakrih/intel/oneapi/mkl/2023.2.0/lib/intel64/;/home/albakrih/intel/oneapi/mkl/2023.2.0/include" ..


# -----Build on Niagara-----------
cmake -DCMAKE_BUILD_TYPE=Release -DPAPI_PREFIX=${HOME}/programs/papi/  ..


# Build (for Make on Unix equivalent to `make -j $(nproc)`)
cmake --build . --config Release -- -j4



