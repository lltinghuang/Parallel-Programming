#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

int B, N, d;
float *Q, *K, *V, *O;

// double getTimeStamp() {
//     struct timeval tv;
//     gettimeofday(&tv, NULL);
//     return (double) tv.tv_usec/1000000 + tv.tv_sec;
// }

__global__ void flash_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K, 
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ l,
    float* __restrict__ m,
    const int B,
    const int N,
    const int d,
    const float scale
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x; 
    const int batch_idx = blockIdx.z;
    const int lane_id = threadIdx.x;

    if (row >= N) return;

    const int batch_offset = batch_idx * N * d;
    const int m_l_offset = batch_idx * N;

    extern __shared__ float sram[];
    float* Qi = sram;                                 // [BLOCK_SIZE][d+1]
    float* Kj = (float*)&Qi[BLOCK_SIZE * (d + 1)];   // [BLOCK_SIZE][d+1]
    float* Vj = (float*)&Kj[BLOCK_SIZE * (d + 1)];   // [BLOCK_SIZE][d+1]

    float mi = m[m_l_offset + row];
    float li = l[m_l_offset + row];

    #pragma unroll
    for (int j = 0; j < d; j++) {
        Qi[lane_id * (d + 1) + j] = Q[batch_offset + row * d + j];
    }
    
    #pragma unroll
    for (int j = 0; j < d; j++) {
        O[batch_offset + row * d + j] = 0.0f;
    }

    const int num_tiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE; //Br = Bc = BLOCK_SIZE
    float si[BLOCK_SIZE];

    for (int tile = 0; tile < num_tiles; tile++) {
        const int tile_offset = tile * BLOCK_SIZE;
        float mi_local = -INFINITY;

        if (tile_offset + lane_id < N) {
            #pragma unroll
            for (int j = 0; j < d; j++) {
                Kj[lane_id * (d + 1) + j] = K[batch_offset + (tile_offset + lane_id) * d + j];
                Vj[lane_id * (d + 1) + j] = V[batch_offset + (tile_offset + lane_id) * d + j];
            }
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE && (tile_offset + k) < N; k++) {
            float qk = 0.0f;
            #pragma unroll
            for (int j = 0; j < d; j++) {
                qk += Qi[lane_id * (d + 1) + j] * Kj[k * (d + 1) + j];
            }
            qk *= scale;
            si[k] = qk;
            mi_local = fmaxf(mi_local, qk);
        }

        
        float mi_new = fmaxf(mi, mi_local);
        
        // compute exp(si - mi_new) and li_local
        float li_local = 0.0f;
        for (int k = 0; k < BLOCK_SIZE && (tile_offset + k) < N; k++) {
            si[k] = expf(si[k] - mi_new);
            li_local += si[k];
        }

        float li_new = expf(mi - mi_new) * li + li_local;
        
        float scale_prev = (li * expf(mi - mi_new)) / li_new;
        float scale_curr = 1.0f / li_new;

        #pragma unroll
        for (int j = 0; j < d; j++) {
            float oj_prev = O[batch_offset + row * d + j];
            float oj_curr = 0.0f;
            
            for (int k = 0; k < BLOCK_SIZE && (tile_offset + k) < N; k++) {
                oj_curr += si[k] * Vj[k * (d + 1) + j];
            }
            
            O[batch_offset + row * d + j] = 
                scale_prev * oj_prev + scale_curr * oj_curr;
        }

        // update mi, li
        mi = mi_new;
        li = li_new;
        __syncthreads();
    }

    m[m_l_offset + row] = mi;
    l[m_l_offset + row] = li;
}

void flash_attention(float *q, float *k, float *v, float *o) {
    float *d_q, *d_k, *d_v, *d_o;
    float *d_m, *d_l;
    
    cudaMalloc(&d_q, B * N * d * sizeof(float));
    cudaMalloc(&d_k, B * N * d * sizeof(float));
    cudaMalloc(&d_v, B * N * d * sizeof(float));
    cudaMalloc(&d_o, B * N * d * sizeof(float));
    cudaMalloc(&d_m, B * N * sizeof(float));
    cudaMalloc(&d_l, B * N * sizeof(float));

    float *h_m = (float*)malloc(B * N * sizeof(float));
    float *h_l = (float*)malloc(B * N * sizeof(float));
    for (int i = 0; i < B * N; i++) {
        h_m[i] = -INFINITY;
        h_l[i] = 0.0f;
    }

    cudaMemcpy(d_q, q, B * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k, B * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, B * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m, B * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, h_l, B * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_o, 0, B * N * d * sizeof(float));

    const int sram_size = (BLOCK_SIZE * (d + 1) + BLOCK_SIZE * (d + 1) + BLOCK_SIZE * (d + 1)) * sizeof(float);

    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, B);  // (rows, 1, batches)
    dim3 block(BLOCK_SIZE);              

    float scale = 1.0f / sqrt(d);
    flash_attention_kernel<<<grid, block, sram_size>>>(
        d_q, d_k, d_v, d_o, d_l, d_m,
        B, N, d, scale
    );

    cudaMemcpy(o, d_o, B * N * d * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
    cudaFree(d_m);
    cudaFree(d_l);
    free(h_m);
    free(h_l);
}

void input(char *input_filename) {
    FILE *file = fopen(input_filename, "rb");
    
    fread(&B, sizeof(int), 1, file); //each batch has Q, K, V
    fread(&N, sizeof(int), 1, file); //Q (K/V): N x d floating point number
    fread(&d, sizeof(int), 1, file);
    
    Q = (float *)malloc(B * N * d * sizeof(float));
    K = (float *)malloc(B * N * d * sizeof(float));
    V = (float *)malloc(B * N * d * sizeof(float));
    O = (float *)malloc(B * N * d * sizeof(float));
    
    for (int i = 0; i < B; i++) {
        fread(Q + (i * N * d), sizeof(float), N * d, file);
        fread(K + (i * N * d), sizeof(float), N * d, file);
        fread(V + (i * N * d), sizeof(float), N * d, file);
    }
    memset(O, 0x00, B * N * d * sizeof(float));
    
    fclose(file);
}

void output(char *output_filename) {
    FILE *file = fopen(output_filename, "wb");
    fwrite(O, sizeof(float), B * N * d, file);
    free(Q);
    free(K);
    free(V);
    free(O);
    fclose(file);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }

    input(argv[1]);

    // double start, end;
    // start = getTimeStamp();

    flash_attention(Q, K, V, O);

    // end = getTimeStamp();
    // printf("(B, N, d): (%d, %d, %d)\n", B, N, d);
    // printf("Time: %.3f seconds\n", end - start);

    output(argv[2]);

    return 0;
}