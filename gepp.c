#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

void print_matrix(double** T, int rows, int cols);
void gepp_basic(double** a, int n);
void print_vector_matrix(double* T, int rows, int cols);
inline double get(double* T, int n, int rows, int cols);

int main(int argc, char* argv[]) {
    double** a;
    double** a1;

    int i, j;
    int n;  // input size
    int b;  // block size

    struct timeval start_time, end_time;
    long seconds, microseconds;
    double elapsed;

    int rank, numprocs;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    if (rank == 0) {
        if (argc == 3) {
            n = atoi(argv[1]);
            b = atoi(argv[2]);
            printf("The matrix size: %d * %d\n", n, n);
            printf("The block size: %d\n\n", b);
        } else {
            printf(
                "Usage: %s n b\n"
                " n: the matrix size\n"
                " b: the block size\n\n",
                argv[0]);
            exit(1);
        }
    }

    n = atoi(argv[1]);
    b = atoi(argv[2]);

    if (rank == 0) {
        printf("Creating and initializing matrices...\n");
        /*** Allocate contiguous memory for 2D matrices ***/
        a = (double**)malloc(n * sizeof(double*));
        a1 = (double**)malloc(n * sizeof(double*));

        double* a0 = (double*)malloc(n * n * sizeof(double));
        double* a10 = (double*)malloc(n * n * sizeof(double));

        for (i = 0; i < n; i++) {
            a[i] = a0 + i * n;
            a1[i] = a10 + i * n;
        }
        printf("Done!\n\n");

        srand(time(0));
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                a1[i][j] = a[i][j] = (double)rand() / RAND_MAX;
            }
        }
    }

    if (rank == 0) {
        printf("Starting sequential computation...\n");
        /**** Sequential computation *****/
        gettimeofday(&start_time, 0);
        gepp_basic(a1, n);
        gettimeofday(&end_time, 0);

        // print the running time
        seconds = end_time.tv_sec - start_time.tv_sec;
        microseconds = end_time.tv_usec - start_time.tv_usec;
        elapsed = seconds + 1e-6 * microseconds;
        printf("sequential calculation time: %f\n\n", elapsed);
    }

    if (rank == 0) {
        // print_matrix(a, n, n);
        printf("Starting MPI computation...\n");
        gettimeofday(&start_time, 0);
    }

    /**** MPI computation *****/
    MPI_Status status;

    const int BLOCK_NUMS = n / b + (n % b == 0 ? 0 : 1);
    const int GROUP_NUMS = BLOCK_NUMS / numprocs + (BLOCK_NUMS % numprocs == 0 ? 0 : 1);

    MPI_Datatype col_t;
    MPI_Datatype process_col_t;
    MPI_Type_vector(n, 1, n, MPI_DOUBLE, &col_t);
    MPI_Type_vector(n, 1, GROUP_NUMS * b, MPI_DOUBLE, &process_col_t);

    MPI_Type_create_resized(col_t, 0, sizeof(double), &col_t);
    MPI_Type_create_resized(process_col_t, 0, sizeof(double), &process_col_t);
    MPI_Type_commit(&col_t);
    MPI_Type_commit(&process_col_t);

    // align with group nums
    int* sendcounts = (int*)malloc(GROUP_NUMS * numprocs * sizeof(int));
    int* displs = (int*)malloc(GROUP_NUMS * numprocs * sizeof(int));

    // initilize sendcounts and displs
    for (i = 0; i < GROUP_NUMS * numprocs; i++) {
        sendcounts[i] = 0;
        displs[i] = 0;
    }

    int remainder = n % b;
    for (i = 0; i < BLOCK_NUMS; i++) {
        sendcounts[i] = b;
    }

    if (remainder != 0) {
        sendcounts[BLOCK_NUMS - 1] = remainder;
    }

    int COL_NUMS = 0;
    for (i = rank; i < BLOCK_NUMS; i += numprocs) {
        COL_NUMS += sendcounts[i];
    }

    for (i = 0; i < BLOCK_NUMS; i++) {
        displs[i] = i * b;
    }

    // double* process = (double*)malloc(n * GROUP_NUMS * b * sizeof(double));

    double** process = malloc(n * sizeof(double*));
    double* process0 = malloc(GROUP_NUMS * b * n * sizeof(double));

    for (i = 0; i < n; i++) {
        process[i] = process0 + i * GROUP_NUMS * b;
    }

    // print
    // if (rank == 0) {
    // 	printf("sendcounts: ");
    // 	for (i = 0; i < GROUP_NUMS * numprocs; i++) {
    // 		printf("%d ", sendcounts[i]);
    // 	}
    // 	printf("\n");

    // 	printf("displs: ");
    // 	for (i = 0; i < GROUP_NUMS * numprocs; i++) {
    // 		printf("%d ", displs[i]);
    // 	}
    // 	printf("\n");
    // }

    // spread data
    for (i = 0; i < GROUP_NUMS; i++) {
        MPI_Scatter(a[0] + i * b * numprocs, b, col_t, process[0] + i * b, b, process_col_t, 0, MPI_COMM_WORLD);
    }

    // for remains
    if (i * b * numprocs < n) {
        MPI_Scatterv(a[0] + i * b * numprocs, sendcounts + i * numprocs, displs + i * numprocs, col_t,
                     process + i * b, b, process_col_t, 0, MPI_COMM_WORLD);
    }

    // printf("rank: %d\n", rank);
    // print_matrix(process, n, COL_NUMS);

    int amax, idx, k, cur;
    int END, BEGIN;
    double* c;

    // main gepp
    // for (int ib = 0; ib < n - b; ib += b) {
    //     END = ib + b;
    //     BEGIN = ib;
    //     if (ib % numprocs == rank) {
    //         for (i = ib; i < END; i++) {
    //             amax = get(process, ib, i - ib, n);
    //             idx = ib;
    //             for (k = ib + 1; k < n; k++) {
    // 				cur = get(process, n, k, i - ib);
    //                 if (fabs(cur) > fabs(amax)) {
    //                     amax = cur;
    //                     idx = k;
    //                 }
    //             }

    //             // exit with a warning that a is singular
    //             if (amax == 0) {
    //                 printf("matrix is singular!\n");
    //                 exit(1);
    //             } else if (idx != i) {
    //                 // swap row i and row k
    //                 for (j = 0; j < n; j++) {
    //                     c = a[i][j];
    //                     a[i][j] = a[indk][j];
    //                     a[indk][j] = c;
    //                 }
    //             }
    //         }
    //     }
    // }

    if (rank == 0) {
        gettimeofday(&end_time, 0);

        // print the running time
        seconds = end_time.tv_sec - start_time.tv_sec;
        microseconds = end_time.tv_usec - start_time.tv_usec;
        elapsed = seconds + 1e-6 * microseconds;
        printf("MPI calculation time: %f\n\n", elapsed);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (fabs(a[i][j] - a1[i][j]) > 1.0E-16) {
                    printf("check fail\n");
                    MPI_Finalize();
                    return 0;
                }
            }
        }
        printf("check pass\n");
    }

    MPI_Finalize();
    return 0;
}

void gepp_basic(double** a, int n) {
    int i, j, k;
    int indk;
    double c, amax;

    for (i = 0; i < n - 1; i++) {
        // find and record k where |a(k,i)|=max|a(j,i)|
        amax = a[i][i];
        indk = i;
        for (k = i + 1; k < n; k++) {
            if (fabs(a[k][i]) > fabs(amax)) {
                amax = a[k][i];
                indk = k;
            }
        }

        // exit with a warning that a is singular
        if (amax == 0) {
            printf("matrix is singular!\n");
            exit(1);
        } else if (indk != i) {
            // swap row i and row k
            for (j = 0; j < n; j++) {
                c = a[i][j];
                a[i][j] = a[indk][j];
                a[indk][j] = c;
            }
        }

        // store multiplier in place of A(k,i)
        for (k = i + 1; k < n; k++) {
            a[k][i] = a[k][i] / a[i][i];
        }

        // subtract multiple of row a(i,:) to zero out a(j,i)
        for (k = i + 1; k < n; k++) {
            c = a[k][i];
            for (j = i + 1; j < n; j++) {
                a[k][j] -= c * a[i][j];
            }
        }
    }
}

void print_matrix(double** T, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f\t", T[i][j]);
        }
        printf("\n");
    }
    printf("\n\n");
    return;
}

void print_vector_matrix(double* T, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f\t", T[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n\n");
    return;
}

double get(double* T, int n, int rows, int cols) { return T[n * cols + rows]; }