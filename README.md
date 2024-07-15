# USYD 2024 COMP5426 Assignment2

a parallel design for Gaussian Elimination in c with mpi

## useage

```sh
make

# run mpi version
# mpirun -np <nunproc> gepp <size> <block>
# example

mpirun -np 4 gepp 1024 8

# run mpi unrolling version
# mpirun -np <nunproc> gepp_u <size>
# example

mpirun -np 4 gepp_u 1024

```