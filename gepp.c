#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

void print_matrix(double** T, int rows, int cols);
void gepp_basic(double** a, int n);
void print_vector_matrix(double* T, int rows, int cols);

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

        double fixed[11][11] = {{0.31, 0.70, 0.59, 0.30, 0.55, 0.77, 0.48, 0.73, 0.07, 0.38, 0.19},
                                {0.83, 0.98, 0.86, 0.16, 0.60, 0.95, 0.29, 0.53, 0.46, 0.69, 0.61},
                                {0.88, 0.83, 0.80, 0.02, 0.05, 0.17, 0.24, 0.64, 0.15, 0.55, 0.33},
                                {0.75, 0.85, 0.88, 0.52, 0.32, 0.61, 0.59, 0.70, 0.80, 0.42, 0.69},
                                {0.66, 0.58, 0.29, 0.61, 0.87, 0.82, 0.07, 0.56, 0.43, 0.95, 0.40},
                                {0.23, 0.98, 0.44, 0.40, 0.21, 0.08, 0.56, 0.76, 0.41, 0.31, 0.61},
                                {0.29, 0.83, 0.93, 0.90, 0.42, 0.64, 0.70, 0.84, 0.33, 0.36, 0.42},
                                {0.61, 0.97, 0.29, 0.43, 0.04, 0.86, 0.87, 0.99, 0.25, 0.10, 0.97},
                                {0.70, 0.50, 0.19, 0.78, 0.06, 0.95, 0.19, 0.36, 0.56, 0.48, 0.19},
                                {0.49, 0.38, 0.61, 0.13, 0.09, 0.45, 0.46, 0.45, 0.87, 0.07, 0.42},
                                {0.17, 0.50, 0.46, 0.03, 0.37, 0.45, 0.28, 0.47, 0.43, 0.98, 0.97}};

        srand(time(0));
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                // a1[i][j] = a[i][j] = (double)rand() / RAND_MAX;
                a1[i][j] = a[i][j] = fixed[i][j];
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
        print_matrix(a, n, n);
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
    // int* sendcounts = (int*)malloc(GROUP_NUMS * numprocs * sizeof(int));
    // int* displs = (int*)malloc(GROUP_NUMS * numprocs * sizeof(int));

    // initilize sendcounts and displs
    // for (i = 0; i < GROUP_NUMS * numprocs; i++) {
    //     sendcounts[i] = 0;
    //     displs[i] = 0;
    // }

    // for (i = 0; i < BLOCK_NUMS; i++) {
    //     sendcounts[i] = b;
    // }

    // if (n % b != 0) {
    //     sendcounts[BLOCK_NUMS - 1] = n % b;
    // }

    // calculate col number
    // basic + remain_block + remain_non_block
    int COL_NUMS = (GROUP_NUMS - (BLOCK_NUMS % numprocs != 0 || n % b != 0)) * b +
                   ((BLOCK_NUMS - (n % b != 0)) % numprocs > rank ? b : 0) +
                   ((BLOCK_NUMS - (n % b != 0)) % numprocs == rank ? n % b : 0);

    // for (i = 0; i < BLOCK_NUMS; i++) {
    //     displs[i] = i * b;
    // }

    // double* process = (double*)malloc(n * GROUP_NUMS * b * sizeof(double));

    double** process = malloc(n * sizeof(double*));
    double* process0 = malloc(GROUP_NUMS * b * n * sizeof(double));

    for (i = 0; i < n; i++) {
        process[i] = process0 + i * GROUP_NUMS * b;
    }

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
    MPI_Request* request = (MPI_Request*)malloc(GROUP_NUMS * sizeof(MPI_Request));

    for (i = 0; i < GROUP_NUMS; i++) {
        MPI_Iscatter(a[0] + i * b * numprocs, b, col_t, process[0] + i * b, b, process_col_t, 0, MPI_COMM_WORLD,
                     request + i);
    }

    // for remains
    // if (i * b * numprocs < n) {
    //     MPI_Scatterv(a[0] + i * b * numprocs, sendcounts + i * numprocs, displs + i * numprocs, col_t, process + i *
    //     b,
    //                  b, process_col_t, 0, MPI_COMM_WORLD);
    // }

    MPI_Waitall(GROUP_NUMS, request, MPI_STATUSES_IGNORE);

    // printf("original\n");
    // printf("rank: %d\n", rank);
    // print_matrix(process, n, COL_NUMS);

    double amax, cur, top, left;
    int idx, k, col;
    int END, BEGIN, PROCESS_BEGIN, LL_RIGHT_BEGIN;
    double* c;
    int* change_sequence = (int*)malloc(b * sizeof(int));

    double** LL_matrix = malloc(b * sizeof(double*));
    double* LL0 = malloc(b * b * sizeof(double));

    for (i = 0; i < b; i++) {
        LL_matrix[i] = LL0 + i * b;
    }

    // main gepp
    for (int ib = 0; ib < n; ib += b) {
        // find pivot row k
        PROCESS_BEGIN = ib / b / numprocs * b;
        END = (ib + b) > n ? n : (ib + b);
        BEGIN = ib;
        if ((ib / b) % numprocs == rank) {
            // printf("rank: %d, ib: %d\n", rank, ib);
            // printf("rank: %d, ib: %d, BEGIN: %d, END: %d, PROCESS_BEGIN: %d\n", rank, ib, BEGIN, END, PROCESS_BEGIN);
            for (i = ib; i < END; i++) {
                col = PROCESS_BEGIN + i - ib;
                amax = process[i][col];
                idx = i;
                for (k = i + 1; k < n; k++) {
                    cur = process[k][col];
                    if (fabs(cur) > fabs(amax)) {
                        amax = cur;
                        idx = k;
                    }
                }

                // printf("rank: %d, i: %d, idx: %d, amax: %lf\n", rank, i, idx, amax);

                // exit with a warning that a is singular
                if (amax == 0) {
                    printf("Rank %d: matrix is singular!\n", rank);
                    exit(1);
                } else if (idx != i) {
                    // swap row i and row k
                    c = process[i];
                    process[i] = process[idx];
                    process[idx] = c;
                    change_sequence[col] = idx;
                } else {
                    change_sequence[col] = -1;
                }

                // store multiplier in place of A(k, col)
                top = process[i][col];
                for (k = i + 1; k < n; k++) {
                    process[k][col] /= top;
                }

                // subtract multiple of row a(i,:) to zero out a(j,i)
                for (k = i + 1; k < END; k++) {
                    top = process[i][k - ib];
                    for (j = i + 1; j < n; j++) {
                        process[j][k - ib] -= top * process[j][col];
                    }
                }
            }
        }

        // broadcast swap
        MPI_Bcast(change_sequence, b, MPI_INT, (ib / b) % numprocs, MPI_COMM_WORLD);

        if ((ib / b) % numprocs != rank) {
            // print sequence
            // for (i = 0; i < b; i++) {
            // 	printf("%d ", change_sequence[i]);
            // }
            // printf("\n");
            for (i = 0; i < b; i++) {
                if (change_sequence[i] != -1) {
                    c = process[i + ib];
                    process[i + ib] = process[change_sequence[i]];
                    process[change_sequence[i]] = c;
                }
            }
        }

        // barrier
        // MPI_Barrier(MPI_COMM_WORLD);

        // printf("row swap\n");
        // printf("rank: %d\n", rank);
        // print_matrix(process, n, COL_NUMS);

        // boradcast LL
        int* disps_ll = (int*)malloc(b * sizeof(int));
        int* blocklens_ll = (int*)malloc(b * sizeof(int));

        for (i = 0; i < b; i++) {
            blocklens_ll[i] = i + 1;
            disps_ll[i] = b * i;
        }

        MPI_Datatype LL;
        MPI_Type_indexed(b, blocklens_ll, disps_ll, MPI_DOUBLE, &LL);
        MPI_Type_commit(&LL);

        // fill LL_matrix
        if ((ib / b) % numprocs == rank) {
            for (i = 0; i < b; i++) {
                for (j = 0; j <= i; j++) {
                    LL_matrix[i][j] = process[PROCESS_BEGIN + i][j];
                }
            }
            // printf("rank: %d\n", rank);
            // print_matrix(LL_matrix, b, b);
        }

        MPI_Bcast(LL_matrix[0], 1, LL, (ib / b) % numprocs, MPI_COMM_WORLD);

        // A(ib:end, end+1:n) = LL / A(ib:end, end+1:n)

        // printf("rank: %d\n", rank);
        // print_matrix(LL_matrix, b, b);

        LL_RIGHT_BEGIN = PROCESS_BEGIN;
        if ((ib / b) % numprocs >= rank) {
            LL_RIGHT_BEGIN += b;
        }

		// printf("rank: %d, LL_RIGHT_BEGIN: %d\n", rank, LL_RIGHT_BEGIN);

        for (i = 0; i < b; i++) {
            for (j = ib + i + 1; j < END; j++) {
                left = LL_matrix[j - ib][i];
                // printf("rank: %d, i: %d, j - ib: %d, left: %lf\n", rank, i, j - ib, left);
                for (k = LL_RIGHT_BEGIN; k < COL_NUMS; k++) {
					// printf("rank: %d\n", rank);
					// printf("i j k: %d %d %d\n", i, j, k);
					// printf("process[j][k], left, process[ib + i][k]: %lf %lf %lf\n", process[j][k], left, process[ib + i][k]);
                    process[j][k] -= left * process[ib + i][k];
                }
            }
        }
		
		
    }

    printf("LL Broadcast\n");
    printf("rank: %d\n", rank);
    print_matrix(process, n, COL_NUMS);
	
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
