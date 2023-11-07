# Demo
In this demo the problem of the SpMM is presented.

To run this code on niagara you have to clone the project first the download the submodules
```git
git submodule init
git submodule update
```

then you have to build the code with the command :
```bash
chmod +x build_practice.sh
./build_practice.sh
```

then to run on the computation node you for the GeMM:
```shell
sbatch ./run.sh
```

and for SpMM
```shell
sbatch ./rub_spmm.sh
```