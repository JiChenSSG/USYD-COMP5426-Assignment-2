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
            printf("The block size: %d\n", b);
            printf("The number of processes: %d\n\n", numprocs);
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

    /*** Allocate contiguous memory for 2D matrices ***/
    a = (double**)calloc(n, sizeof(double*));
    a1 = (double**)calloc(n, sizeof(double*));

    double* a0 = (double*)calloc(n * (n + b * numprocs), sizeof(double));
    double* a10 = (double*)calloc(n * n, sizeof(double));

    for (i = 0; i < n; i++) {
        a[i] = a0 + i * (n + b * numprocs);
        a1[i] = a10 + i * n;
    }

    srand(time(0));
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            a1[i][j] = a[i][j] = (double)rand() / RAND_MAX;
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

    MPI_Barrier(MPI_COMM_WORLD);

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
    MPI_Type_vector(n, 1, n + b * numprocs, MPI_DOUBLE, &col_t);
    MPI_Type_vector(n, 1, GROUP_NUMS * b, MPI_DOUBLE, &process_col_t);

    MPI_Type_create_resized(col_t, 0, sizeof(double), &col_t);
    MPI_Type_create_resized(process_col_t, 0, sizeof(double), &process_col_t);
    MPI_Type_commit(&col_t);
    MPI_Type_commit(&process_col_t);

    // calculate col number
    // basic + remain_block + remain_non_block
    int COL_NUMS = (GROUP_NUMS - (BLOCK_NUMS % numprocs != 0 || n % b != 0)) * b +
                   ((BLOCK_NUMS - (n % b != 0)) % numprocs > rank ? b : 0) +
                   ((BLOCK_NUMS - (n % b != 0)) % numprocs == rank ? n % b : 0);

    double** process = (double**)calloc(n, sizeof(double*));
    double* process0 = (double*)calloc(GROUP_NUMS * b * n, sizeof(double));

    for (i = 0; i < n; i++) {
        process[i] = process0 + i * GROUP_NUMS * b;
    }

    // spread data
    MPI_Request* request = (MPI_Request*)calloc(GROUP_NUMS, sizeof(MPI_Request));

    for (i = 0; i < GROUP_NUMS; i++) {
        MPI_Iscatter(a[0] + i * b * numprocs, b, col_t, process[0] + i * b, b, process_col_t, 0, MPI_COMM_WORLD,
                     request + i);
    }

    MPI_Waitall(GROUP_NUMS, request, MPI_STATUSES_IGNORE);

    double amax, cur, top, left;
    int idx, k, col;
    int END, BEGIN, PROCESS_BEGIN, PROCESS_END, LL_RIGHT_BEGIN, CUR_RANK;
    double* c;
    double c0;
    int* change_sequence = (int*)calloc(b, sizeof(int));
    int* disps_ll = (int*)calloc(b, sizeof(int));
    int* blocklens_ll = (int*)calloc(b, sizeof(int));

    double** left_matrix = (double**)calloc(n, sizeof(double));
    double* left0 = (double*)calloc(n * b, sizeof(double));

    for (i = 0; i < n; i++) {
        left_matrix[i] = left0 + i * b;
    }

    double** LL_matrix = (double**)calloc(b, sizeof(double*));
    double* LL0 = (double*)calloc(b * b, sizeof(double));

    for (i = 0; i < b; i++) {
        LL_matrix[i] = LL0 + i * b;
    }

    // main gepp
    for (int ib = 0; ib < n; ib += b) {
        // find pivot row k
        PROCESS_BEGIN = ib / b / numprocs * b;
        PROCESS_END = PROCESS_BEGIN + b > COL_NUMS ? COL_NUMS : PROCESS_BEGIN + b;
        END = (ib + b) > n ? n : (ib + b);
        BEGIN = ib;
        CUR_RANK = (ib / b) % numprocs;

        if (CUR_RANK == rank) {
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

                // exit with a warning that a is singular
                if (amax == 0) {
                    printf("Rank %d: matrix is singular!\n", rank);
                    // exit(1);
                } else if (idx != i) {
                    // swap row i and row k
                    for (j = 0; j < COL_NUMS; j++) {
                        c0 = process[i][j];
                        process[i][j] = process[idx][j];
                        process[idx][j] = c0;
                    }
                    change_sequence[i - ib] = idx;
                } else {
                    change_sequence[i - ib] = -1;
                }

                // store multiplier in place of A(k, col)
                top = process[i][col];
                for (k = i + 1; k < n; k++) {
                    process[k][col] /= top;
                }

                // subtract multiple of row a(i,:) to zero out a(j,i)
                for (k = i + 1; k < END; k++) {
                    top = process[i][PROCESS_BEGIN + k - ib];
                    for (j = i + 1; j < n; j++) {
                        process[j][PROCESS_BEGIN + k - ib] -= top * process[j][col];
                    }
                }
            }
        }

        // broadcast swap
        MPI_Bcast(change_sequence, b, MPI_INT, CUR_RANK, MPI_COMM_WORLD);

        // swap rows in other ranks
        if (CUR_RANK != rank) {
            for (i = ib; i < END; i++) {
                int cs_idx = i - ib;
                if (change_sequence[cs_idx] != -1) {
                    for (j = 0; j < COL_NUMS; j++) {
                        c0 = process[i][j];
                        process[i][j] = process[change_sequence[cs_idx]][j];
                        process[change_sequence[cs_idx]][j] = c0;
                    }
                }
            }
        }

        // boradcast LL
        for (i = 0; i < b; i++) {
            blocklens_ll[i] = i + 1;
            disps_ll[i] = b * i;
        }

        MPI_Datatype LL;
        MPI_Type_indexed(b, blocklens_ll, disps_ll, MPI_DOUBLE, &LL);
        MPI_Type_commit(&LL);

        // fill LL_matrix
        if (CUR_RANK == rank) {
            for (i = PROCESS_BEGIN; i < PROCESS_END; i++) {
                for (j = PROCESS_BEGIN; j <= i; j++) {
                    LL_matrix[i - PROCESS_BEGIN][j - PROCESS_BEGIN] = process[i + ib - PROCESS_BEGIN][j];
                }
            }
        }

        MPI_Bcast(LL_matrix[0], 1, LL, (ib / b) % numprocs, MPI_COMM_WORLD);

        // A(ib:end, end+1:n) = LL / A(ib:end, end+1:n)
        LL_RIGHT_BEGIN = PROCESS_BEGIN;
        if (CUR_RANK >= rank) {
            LL_RIGHT_BEGIN += b;
        }

        for (i = 0; i < b; i++) {
            for (j = ib + i + 1; j < END; j++) {
                left = LL_matrix[j - ib][i];
                for (k = LL_RIGHT_BEGIN; k < COL_NUMS; k++) {
                    process[j][k] -= left * process[ib + i][k];
                }
            }
        }

        // bcast left matrix
        int LEN_REMAIN = n - END;
        if (CUR_RANK == rank) {
            for (i = END; i < n; i++) {
                for (j = PROCESS_BEGIN; j < PROCESS_END; j++) {
                    left_matrix[i][j - PROCESS_BEGIN] = process[i][j];
                }
            }
        }

        MPI_Bcast(left_matrix[END], LEN_REMAIN * b, MPI_DOUBLE, CUR_RANK, MPI_COMM_WORLD);

        // calculate trail matrix
        // printf("rank: %d\n", rank);
        for (i = LL_RIGHT_BEGIN; i < COL_NUMS; i++) {
            for (j = END; j < n; j++) {
                for (k = 0; k < b; k++) {
                    process[j][i] -= left_matrix[j][k] * process[ib + k][i];
                }
            }
        }
    }

    // gather
    for (i = 0; i < GROUP_NUMS; i++) {
        MPI_Igather(process[0] + i * b, b, process_col_t, a[0] + i * b * numprocs, b, col_t, 0, MPI_COMM_WORLD,
                    request + i);
    }

    MPI_Waitall(GROUP_NUMS, request, MPI_STATUSES_IGNORE);

    if (rank == 0) {
        gettimeofday(&end_time, 0);

        // print_matrix(a1, n, n);
        // print_matrix(a, n, n);

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
