//TILENUMd version: 41, 241.05 
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__device__ const int INF = ((1 << 30) - 1);
__device__ const int BLOCKSIZE = 64;    //block size   
__device__ const int TILENUM = 2;       //TILENUM x TILENUM sub-block (for each thread)
__device__ const int TBSIZE = 32;       //thread block size: number of threads per block

int n, m;          // original size
int padding_n;     // padded size
int* host_D;

void Input(const char* inFile) {
    FILE* file = fopen(inFile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    padding_n = n + (BLOCKSIZE - (n % BLOCKSIZE));

    host_D = (int*) malloc(padding_n * padding_n * sizeof(int));
    
    for (int i = 0; i < padding_n; ++i) {
        for (int j = 0; j < padding_n; ++j) {
            host_D[i * padding_n + j] = (i == j) ? 0 : INF;
        }
    }

    int* pairs = (int*)malloc(3 * m * sizeof(int));
    fread(pairs, sizeof(int), 3 * m, file);
    for (int i = 0; i < 3 * m; i += 3) {
        host_D[pairs[i] * padding_n + pairs[i + 1]] = pairs[i + 2];
    }
    free(pairs);
    fclose(file);
}

void Output(const char* outFile) {
    FILE* outfile = fopen(outFile, "w");
    for (int i = 0; i < n; ++i) {
        fwrite(&host_D[i * padding_n], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

__global__ void block_FW_phase1(int* d_D, const int n, const int round) {
    __shared__ int s_D[BLOCKSIZE][BLOCKSIZE];
    
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    
    const int start_x = tid_x * TILENUM;
    const int start_y = tid_y * TILENUM;
    
    const int pivot_base = round * BLOCKSIZE;
    
    #pragma unroll TILENUM
    for(int i = 0; i < TILENUM; i++) {
        for(int j = 0; j < TILENUM; j++) {
            const int global_x = pivot_base + start_x + j;
            const int global_y = pivot_base + start_y + i;
            s_D[start_y + i][start_x + j] = d_D[global_y * n + global_x];
        }
    }
    
    __syncthreads();
    
    #pragma unroll BLOCKSIZE
    for(int k = 0; k < BLOCKSIZE; k++) {
        for(int i = 0; i < TILENUM; i++) {
            for(int j = 0; j < TILENUM; j++) {
                const int idx_i = start_y + i;
                const int idx_j = start_x + j;
                s_D[idx_i][idx_j] = min(
                    s_D[idx_i][idx_j],
                    s_D[idx_i][k] + s_D[k][idx_j]
                );
            }
        }
    }

    #pragma unroll TILENUM
    for(int i = 0; i < TILENUM; i++) {
        for(int j = 0; j < TILENUM; j++) {
            const int global_x = pivot_base + start_x + j;
            const int global_y = pivot_base + start_y + i;
            d_D[global_y * n + global_x] = s_D[start_y + i][start_x + j];
        }
    }
}

__global__ void block_FW_phase2(int* d_D, const int n, const int round) {
    if(blockIdx.x == round) return;
    
    __shared__ int s_row[BLOCKSIZE][BLOCKSIZE];
    __shared__ int s_col[BLOCKSIZE][BLOCKSIZE];
    
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    
    const int start_x = tid_x * TILENUM;
    const int start_y = tid_y * TILENUM;
    
    const int pivot_base = round * BLOCKSIZE;
    
    register int result_row[TILENUM][TILENUM];
    register int result_col[TILENUM][TILENUM];
    
    #pragma unroll TILENUM
    for(int bi = 0; bi < TILENUM; bi++) {
        for(int bj = 0; bj < TILENUM; bj++) {
            int i = threadIdx.y + bi * TBSIZE;
            int j = threadIdx.x + bj * TBSIZE;
            
            int global_i = blockIdx.x * BLOCKSIZE + i;
            int global_j = pivot_base + j;
            
            result_row[bi][bj] = d_D[global_i * n + global_j];
            s_row[i][j] = result_row[bi][bj];
            
            result_col[bi][bj] = d_D[(pivot_base + i) * n + blockIdx.x * BLOCKSIZE + j];
            s_col[i][j] = result_col[bi][bj];
        }
    }
    
    __syncthreads();
    
    for(int k = 0; k < BLOCKSIZE; k++) {
        for(int bi = 0; bi < TILENUM; bi++) {
            for(int bj = 0; bj < TILENUM; bj++) {
                int i = threadIdx.y + bi * TBSIZE;
                int j = threadIdx.x + bj * TBSIZE;
                
                result_row[bi][bj] = min(
                    result_row[bi][bj],
                    s_row[i][k] + d_D[(pivot_base + k) * n + pivot_base + j]
                );
                
                result_col[bi][bj] = min(
                    result_col[bi][bj],
                    d_D[(pivot_base + i) * n + pivot_base + k] + s_col[k][j]
                );
            }
        }
    }
    
    #pragma unroll TILENUM
    for(int bi = 0; bi < TILENUM; bi++) {
        for(int bj = 0; bj < TILENUM; bj++) {
            int i = threadIdx.y + bi * TBSIZE;
            int j = threadIdx.x + bj * TBSIZE;
            
            int global_i = blockIdx.x * BLOCKSIZE + i;
            int global_j = pivot_base + j;
            
            d_D[global_i * n + global_j] = result_row[bi][bj];
            d_D[(pivot_base + i) * n + blockIdx.x * BLOCKSIZE + j] = result_col[bi][bj];
        }
    }
}

__global__ void block_FW_phase3(int* D, const int n, const int round) {
    if(blockIdx.x == round || blockIdx.y == round) return;
    
    __shared__ int s_row[BLOCKSIZE][BLOCKSIZE];
    __shared__ int s_col[BLOCKSIZE][BLOCKSIZE];
    register int s_remaining[TILENUM][TILENUM];
    
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    
    const int start_x = tid_x * TILENUM;
    const int start_y = tid_y * TILENUM;

    const int pivot_base = round * BLOCKSIZE;
    
    #pragma unroll TILENUM
    for(int i = 0; i < TILENUM; i++) {
        for(int j = 0; j < TILENUM; j++) {
            const int row_x = blockIdx.x * BLOCKSIZE + start_x + j;
            const int row_y = pivot_base + start_y + i;
            s_row[start_y + i][start_x + j] = D[row_y * n + row_x];
        }
    }
    
    #pragma unroll TILENUM
    for(int i = 0; i < TILENUM; i++) {
        for(int j = 0; j < TILENUM; j++) {
            const int col_x = pivot_base + start_x + j;
            const int col_y = blockIdx.y * BLOCKSIZE + start_y + i;
            s_col[start_y + i][start_x + j] = D[col_y * n + col_x];
        }
    }
    
    #pragma unroll TILENUM
    for(int i = 0; i < TILENUM; i++) {
        for(int j = 0; j < TILENUM; j++) {
            const int global_x = BLOCKSIZE * blockIdx.x  + start_x + j;
            const int global_y = BLOCKSIZE * blockIdx.y + start_y + i;
            s_remaining[i][j] = D[global_y * n + global_x];
        }
    }
    __syncthreads();
    
    for (int k = 0; k < BLOCKSIZE; ++k) {
        for(int i = 0; i < TILENUM; i++) {
            for(int j = 0; j < TILENUM; j++) {
                    s_remaining[i][j] = min(
                        s_remaining[i][j],
                        s_col[start_y + i][k] + s_row[k][start_x + j]
                    );
            }
        }
    }
    
    #pragma unroll TILENUM
    for(int i = 0; i < TILENUM; i++) {
        for(int j = 0; j < TILENUM; j++) {
            const int global_x = BLOCKSIZE * blockIdx.x + start_x + j;
            const int global_y = BLOCKSIZE * blockIdx.y + start_y + i;
            D[global_y * n + global_x] = s_remaining[i][j];
        }
    }
}

int main(int argc, char** argv) {
    Input(argv[1]);
    
    int* dev_D;
    size_t total_size = padding_n * padding_n * sizeof(int);
    cudaMalloc(&dev_D, total_size);
    cudaMemcpy(dev_D, host_D, total_size, cudaMemcpyHostToDevice);
    
    // Set grid and block dimensions
    const int round = padding_n / BLOCKSIZE;
    dim3 block_dim(TBSIZE, TBSIZE);
    dim3 grid_dim(round, round);

    for (int r = 0; r < round; r++) {
        block_FW_phase1<<<1, block_dim>>>(dev_D, padding_n, r);
        block_FW_phase2<<<round, block_dim>>>(dev_D, padding_n, r);
        block_FW_phase3<<<grid_dim, block_dim>>>(dev_D, padding_n, r);
    }
    
    cudaMemcpy(host_D, dev_D, total_size, cudaMemcpyDeviceToHost);
    
    Output(argv[2]);
    
    cudaFree(dev_D);
    free(host_D);
    
    return 0;
}