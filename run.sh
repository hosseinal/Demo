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
module load mkl/2022u2
module load cmake
module load gcc


echo "---- running an example ----"

cd build

rm ../results.csv

# shellcheck disable=SC2034
#size of the matrix
sizes="64 128 256 512"
# shellcheck disable=SC2034
thread_list="1"

# shellcheck disable=SC2034
HEADER="ON"
BLOCKSIZE=16
for size in $sizes; do
  if [ "$size" -lt 64 ]; then
    BLOCKSIZE=16
  else
    BLOCKSIZE=64
  fi
    echo $BLOCKSIZE

  # shellcheck disable=SC2034
  for thr in $thread_list;do
  export MKL_NUM_THREADS=$thr
  if [ "$HEADER" == "ON" ]; then
    ./DemoGeMM "$thr" "$size" 1 "$BLOCKSIZE"
    HEADER="OFF"
  else
    ./DemoGeMM "$thr" "$size" 0 "$BLOCKSIZE"
  fi
  done
done
done
