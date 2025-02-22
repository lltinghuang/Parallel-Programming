# Parallel Programming HW4

## Implementation
a. Describe how you implemented the FlashAttention forward pass using CUDA. Mention the algorithm's key steps, such as matrix blocking, SRAM usage, and how intermediate results like scaling factors (*l* and *m*) were calculated.
- Matrix blocking
    在FlashAttention中利用了tiling的技巧來降低計算複雜度跟記憶體存取的開銷，我讓每個cuda block處理BLOCK_SIZE行的Q，然後跟K和V的BLOCK_SIZE列進行乘積計算。
    num_tiles表示Key和Value的列被分成幾個tiles，Q的tile數量 = K的tile數量 = V的tile數量 = $\lceil \frac{N}{BLOCK\_SIZE} \rceil$
    
- SRAM usage
    將$Q_i$、$K_j$和$V_j$ (小塊的資料)分別載入shared memory，然後在shared memory中計算attention score $S_{ij}$和Softmax更新。
    ```c=
    extern __shared__ float sram[];
    float* Qi = sram;                     //Br*d的Query block
    float* Kj = (float*)&Qi[Br * (d + 1)];//Bc*d的Key block
    float* Vj = (float*)&Kj[Bc * (d + 1)];//Bc*d的Value block
    float* S  = (float*)&Vj[Bc * (d + 1)];//Temporary buffer for s_ij
    ```
    $Q_i$、$K_j$和$V_j$ 分配的大小均為$BLOCK\_SIZE * d$，額外的+1是為了避免bank conflict所做的padding。

- Calculation for scaling factor
    *m* 用來記錄最大值，後續用於穩定softmax的計算以避免overflow，*l* 用於對softmax輸出進行normalize。
    ```c=
    float mi_local = -INFINITY;
    for (int k = 0; k < BLOCK_SIZE && (tile_offset + k) < N; k++) {
        float qk = 0.0f;
        #pragma unroll
        for (int j = 0; j < d; j++) {
            qk += Qi[lane_id * (d + 1) + j] * Kj[k * (d + 1) + j];
        }
        qk *= scale; // 縮放注意力分數
        si[k] = qk;
        mi_local = fmaxf(mi_local, qk); // 記錄當前分塊的最大值
    }
    ```
    然後將當前分塊(tile)的局部最大值跟當前行的全局最大值合併來得到更新的$m_{new}$，這樣設計能確保整行 $QK^T$ 的最大值是逐步累積的，能避免數值不穩定的狀況。
    ```c=
    float li_local = 0.0f;
    for (int k = 0; k < BLOCK_SIZE && (tile_offset + k) < N; k++) {
        si[k] = expf(si[k] - mi_new); 
        li_local += si[k]; // 累加分塊的 softmax 結果
    }
    ```
    先使用*m*穩定化$S_{ij}$，計算 $exp(S_{ij} - m)$並累加。
    更新 *l* ：
    `float li_new = expf(mi - mi_new) * li + li_local;`
    將之前累積的 *l* 根據新的最大值 *m* 來對當前分塊的softmax結果(li_local)進行縮放。最後再將更新後的 *l* 和 *m* 寫回global memory，供下次的計算使用。
    
- Kernel Execution
    在單一kernel內完成所有計算，grid分配N行Query，每個block分配32個thread。
    ```c=
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, B);  // (rows, 1, batches)
    dim3 block(BLOCK_SIZE);

    float scale = 1.0f / sqrt(d);
    flash_attention_kernel<<<grid, block, sram_size>>>(d_q, d_k, d_v, d_o, d_l, d_m,B, N, d, scale);
    ```
    
    主循環為處理所有的Key和Value block，對於每個block，先載入所需的Q, K, V到shared memory，然後依序計算$QK^T$、softmax、輸出結果，以及更新 *m* 和 *l*，直到所有的Key和Value block處理完畢。
    
   ```c=
    for (int j = 0; j < d; j++) {
        Qi[lane_id * (d + 1) + j] = Q[batch_offset + row * d + j];
    }
    ```
    每個thread只要載入和處理自己負責的那行Query，而shared memory中的$Q_i$可以被同個block的thread重複使用，當每個thread負責計算$S_{ij} = Q_i K_j^T$的時候，就不需要再access global memory。
    ```c=
    for (int k = 0; k < BLOCK_SIZE && (tile_offset + k) < N; k++) {
        float qk = 0.0f;
        #pragma unroll
        for (int j = 0; j < d; j++) {
            qk += Qi[lane_id * (d + 1) + j] * Kj[k * (d + 1) + j];
        }
        qk *= scale;
        si[k] = qk;
    }
    ```


b. Explain how matrices Q, K, and V are divided into blocks and processed in parallel.

每個block處理block size條Query，而K和V被進一步分成$BLOCK\_SIZE * d$的tile，然後做分塊操作。
每個block負責一個tile，每個thread處理一行Q，使用ThreadIdx.x平行計算tile的內容。

c. Describe how you chose the block sizes B_r and B_c and why.
讓$B_r = B_c = BLOCK\_SIZE = 32$，$B_r$是每次處理的Query行數，$B_c$是每次處理的Key/Value的列數。設為32是因為這樣能充分利用GPU的warp (每個warp有32個thread)，這樣子可以讓warp高效執行。

d. Specify the configurations for CUDA kernel launches, such as the number of threads per block, shared memory allocation, and grid dimensions.

每個block分配block size (32)個thread用來處理每個Query的一行，並且分配$3*BLOCK\_SIZE*(d+1)$的shared memory，用來存放$Q_i, K_j, V_j$的局部資料，然後用(d+1)來避免bank conflict。
因為每行 Query 的計算是相互獨立的，所以可以在X軸將Query的行進行切分，然後分配給不同的cuda block。並且不同block之間的計算完全獨立，也就不需要額外的同步。
    
```c=
const int sram_size = (BLOCK_SIZE * (d + 1) + BLOCK_SIZE * (d + 1) + BLOCK_SIZE * (d + 1)) * sizeof(float);
dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, B);  // (rows, 1, batches)
dim3 block(BLOCK_SIZE);   
float scale = 1.0f / sqrt(d);
flash_attention_kernel<<<grid, block, sram_size>>>(
        d_q, d_k, d_v, d_o, d_l, d_m,
        B, N, d, scale
    );
```


e. Justify your choices and how they relate to the blocking factors and the SRAM size.

因為BLOCK_SIZE=32可以匹配GPU warp的架構，避免thread閒置。而shared memory設為BLOCK_SIZE和d的乘積，能確保他有足夠的儲存空間存放必要的數據，並利用padding (d+1)來避免bank conflict，能有效提升shared memory的效率。

## Profiling Results
測資：t15
![{F4617407-8773-41C5-B712-E7203A70C49A}](https://hackmd.io/_uploads/BycRkYEr1x.png)


## Experiment & Analysis
a. System Spec: apollo-gpu
b. Optimization
測資：t15
![image](https://hackmd.io/_uploads/ByiUi5EHJe.png)

![image](https://hackmd.io/_uploads/SJP8RSHB1l.png)

從以上兩張圖表可以看出，再加入了handle bank conflict的優化之後，有大幅的加速，從Efficiency Performance的結果可以看出，這個優化主要影響了shared_efficiency，從shared memory版本的3.12%提升到36.78%。這是因為本來多個thread可能同時存取同一個記憶體的rank，但是在shared memory分配時加入了padding可以使得每行的數據分布跨越不同的bank，就可以減少thread因為bank conflict而需要sequential執行的情況。

而最後一項優化dynamic grid setting是讓grid的設置根據輸入的N來調整（原本是固定dim grid(32))，從Efficiency Performance的結果可以看出，他主要影響了achieved_occupancy，從unroll版本的2.51%提升到10.92%，這是因為本來的設定可能無法充分利用所有SM，根據輸入的N動態調整grid大小可以更好的分配工作負載，但是10.92%還是偏低，表示GPU資源未被充分利用，或許可以再加入multi-stream的方法來優化。

c. Others

<image src="https://hackmd.io/_uploads/BksE4wSryl.png" width=45%><image src="https://hackmd.io/_uploads/SyKD4vHSkx.png" width=45%>

這個實驗分析了不同輸入參數對於memory throughput的影響，可以看出對於不同的N和B，他們的數值變化對throughput並無太大的影響，然而d=32則比d=64有將近兩倍的throughput。這是因為當d比較小的時候，每次memory access的數據較小，可以更有效地利用快取和記憶體頻寬，減少了memory access的延遲。


