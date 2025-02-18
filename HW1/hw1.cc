// v5: put function into "main" => 93.10
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <math.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <boost/sort/spreadsort/spreadsort.hpp>
#include <omp.h>
using namespace std;

inline int get_partition_size(int rank, int chunk_size, int n, int start_idx) {
    if (rank < 0 || start_idx >= n) {
        return 0; // boundary condition
    }
    if (start_idx + chunk_size < n) {
        return chunk_size; 
    }
    return n - start_idx; 
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = atoi(argv[1]);        // total number of elements
    char *input_filename = argv[2];
    char *output_filename = argv[3];
    int chunk_size = ceil(n/double(size));  // ideal partition size for each process
    int cnt;                      // actual number of elements in current process
    int rcnt;                     // number of elements in right process
    int lcnt;                     // number of elements in left process
    int start_idx;               // starting index of current process

    start_idx = chunk_size* rank;
    cnt = get_partition_size(rank, chunk_size, n, start_idx);
    rcnt = get_partition_size(rank+1, chunk_size, n, start_idx + chunk_size);
    lcnt = get_partition_size(rank-1, chunk_size, n, start_idx - chunk_size);

    float *data = new float[cnt];
    float *buffer = new float[chunk_size];
    float *tmp= new float[cnt];

    MPI_File input_file, output_file;

    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    if(cnt != 0){
        MPI_File_read_at(input_file, sizeof(float) * start_idx, data, cnt, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&input_file);

    if (cnt > 0)
        boost::sort::spreadsort::float_sort(data, data + cnt);

    int pass_cnt = 0;
    bool phase = true;  // 0: odd, 1: even
    bool last = cnt >0 && (start_idx + chunk_size)>= n;  // 是否為最後一個進程
    bool sorted = false;
    MPI_Request send_request, recv_request;
    int i, j, k;

    while (pass_cnt <= size){ 
        if (cnt != 0){
            if ((rank & 1) ^ phase){ //even phase + even rank, odd phase + odd rank
                if (!last && rcnt != 0){
                    // 與右側進程比較和合併
                    //MPI_Sendrecv(&data[cnt - 1], 1, MPI_FLOAT, rank + 1, 0, buffer, 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Isend(&data[cnt - 1], 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &send_request);
                    MPI_Irecv(buffer, 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &recv_request);
                    MPI_Wait(&send_request, MPI_STATUS_IGNORE);
                    MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
                    //當前rank的資料存在data[cnt-1], rank+1的資料存在buffer
                    //當前rank的最大值大於rank+1的最小值的話，要做合併
                    if(data[cnt-1] > buffer[0]){
                        sorted = false;
                        //MPI_Sendrecv(data, cnt, MPI_FLOAT, rank + 1, 0, buffer, rcnt, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Isend(data, cnt, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD, &send_request);
                        MPI_Irecv(buffer, rcnt, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD, &recv_request);
                        MPI_Wait(&send_request, MPI_STATUS_IGNORE);
                        MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
                        //merge with right neighbor
                        i = j = k = 0;

                        for (k = 0; k < cnt; k++) {
                            if(j >= rcnt) tmp[k] = data[i++];
                            else tmp[k] = (data[i] < buffer[j]) ? data[i++] : buffer[j++];
                        }
    
                        #pragma omp parallel for
                        for (i = 0; i < cnt; i++) data[i] = tmp[i];
                        
                    }else{
                        sorted = true;
                    }
                }
            } else{
                if (rank != 0){
                    // 與左側進程比較和合併
                    //MPI_Sendrecv(&data[0], 1, MPI_FLOAT, rank - 1, 0, buffer, 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Isend(&data[0], 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &send_request);
                    MPI_Irecv(buffer, 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &recv_request);
                    MPI_Wait(&send_request, MPI_STATUS_IGNORE);
                    MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
                    if(data[0] < buffer[0]){
                        sorted = false;
                        //MPI_Sendrecv(data, cnt, MPI_FLOAT, rank - 1, 0, buffer, lcnt, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Isend(data, cnt, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD, &send_request);
                        MPI_Irecv(buffer, lcnt, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD, &recv_request);
                        MPI_Wait(&send_request, MPI_STATUS_IGNORE);
                        MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
                        //merge with left neighbor
                        int i, j, k;
                        i = k = cnt - 1;
                        j = lcnt - 1;

                        for (int k = cnt - 1; k >= 0; --k) {
                            if (data[i] > buffer[j]) tmp[k] = data[i--];
                            else tmp[k] = buffer[j--];
                        }
    
                        #pragma omp parallel for
                        for (int i = 0; i < cnt; i++) data[i] = tmp[i];
                    }else{
                        sorted = true;
                    }
                }
            }
        }
        
        phase = !phase;  // 切換奇偶階段
        
        pass_cnt += 1;
    }

    if(cnt != 0){
        MPI_File_write_at(output_file, sizeof(float) * start_idx, data, cnt, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&output_file); 

    delete [] data;
    delete [] buffer;
    delete [] tmp;

    MPI_Finalize();
    return 0;
}