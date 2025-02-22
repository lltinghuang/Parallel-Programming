#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <omp.h>
#include <mpi.h>

#define min(a,b) ((a) < (b) ? (a) : (b))
const int INF = ((1 << 30) - 1);
const int V = 50010;

void input(char* inFileName, int rank);
void output(char* outFileName, int rank);
void block_FW(int B, int rank, int size);
int ceil(int a, int b);
void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);

int n, m;
static int Dist[V][V];

int main(int argc, char* argv[]) {
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    input(argv[1], rank);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for(int i = 0; i < n; i++) {
        MPI_Bcast(Dist[i], 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    int B = 256;
    block_FW(B, rank, size);

    for(int i = 0; i < n; i++) {
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : Dist[i], 
                  Dist[i], n, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    }

    output(argv[2], rank);
    MPI_Finalize();
    return 0;
}

void block_FW(int B, int rank, int size) {
   int round = ceil(n, B);
   int blocks_per_process = (round + size - 1) / size;
   int start_block = rank * blocks_per_process;
   int end_block = min((rank + 1) * blocks_per_process, round);

   for (int r = 0; r < round; ++r) {
       if (rank == 0) {
           cal(B, r, r, r, 1, 1);
       }

       if((r + 1) * B <= n) {
           MPI_Bcast(&Dist[r * B][r * B], 1, MPI_INT, 0, MPI_COMM_WORLD);
       } else {
           for(int i = r * B; i < min((r + 1) * B, n); i++) {
               MPI_Bcast(Dist[i], n, MPI_INT, 0, MPI_COMM_WORLD);
           }
       }

       #pragma omp parallel for schedule(dynamic, 1)
       for (int i = start_block; i < end_block; i++) {
           if (i != r) {
               cal(B, r, i, r, 1, 1);
               cal(B, r, r, i, 1, 1);
           }
       }

       #pragma omp parallel for collapse(2) schedule(dynamic, 1)
       for (int i = start_block; i < end_block; i++) {
           for (int j = 0; j < round; j++) {
               if (i != r || j != r) {
                   cal(B, r, i, j, 1, 1);
               }
           }
       }
   }
}

void cal(int B, int Round, int block_start_x, int block_start_y,
         int block_width, int block_height) {
    int block_internal_start_x = block_start_x * B;
    int block_internal_end_x = min((block_start_x + block_height) * B, n);
    int block_internal_start_y = block_start_y * B;
    int block_internal_end_y = min((block_start_y + block_width) * B, n);

    for (int k = Round * B; k < min((Round + 1) * B, n); k++) {
        // #pragma omp parallel for schedule(dynamic, 32)
        for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
            if (Dist[i][k] == INF) continue;
            // #pragma omp simd
            for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
                if (Dist[k][j] != INF) {
                    int new_dist = Dist[i][k] + Dist[k][j];
                    if (new_dist < Dist[i][j]) {
                        Dist[i][j] = new_dist;
                    }
                }
            }
        }
    }
}

void input(char* infile, int rank) {
    if (rank == 0) {
        FILE* file = fopen(infile, "rb");
        fread(&n, sizeof(int), 1, file);
        fread(&m, sizeof(int), 1, file);
        // printf("n: %d, m: %d\n", n, m);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == j) {
                    Dist[i][j] = 0;
                } else {
                    Dist[i][j] = INF;
                }
            }
        }

        int pair[3];
        for (int i = 0; i < m; ++i) {
            fread(pair, sizeof(int), 3, file);
            Dist[pair[0]][pair[1]] = pair[2];
        }
        fclose(file);
    }
}

void output(char* outFileName, int rank) {
    if (rank == 0) {
        FILE* outfile = fopen(outFileName, "w");
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (Dist[i][j] >= INF) Dist[i][j] = INF;
            }
            fwrite(Dist[i], sizeof(int), n, outfile);
        }
        fclose(outfile);
    }
}

int ceil(int a, int b) { 
    return (a + b - 1) / b; 
}