# Parallel Programming HW3

## Implementation
### 3-1
實現的版本：Blocked Floyd-Warshall algorithm 

一開始用input讀取輸入的節點數量n，然後初始化 n x n 的Dist矩陣，所有節點的距離為無限大，到自己的距離設為0，以及讀入m個edge的資料(節點i到節點j的距離為w，把它記為Dist[i][j]=w)。而這裡只讓rank=0讀取資料，再用MPI_Bcast廣播給其他MPI節點。

block_FW(B, rank, size)執行Blocked Floyd-Warshall的演算法。
![{2D102DE5-69FD-485C-BA68-72259702D640}](https://hackmd.io/_uploads/Sk-nK-Pryg.png)

演算法概念如spec的示意圖，總共會做round次，因為我們切分的block大小為block size (B)，所以$round = \lceil \frac{n}{B} \rceil$。
每次會分別計算
- Phase 1：Pivot block
    會由rank = 0處理pivot block，然後透過MPI_Bcast廣播給其他節點。
    ```cpp
    if (rank == 0) {
        cal(B, r, r, r, 1, 1);
    }
    MPI_Bcast(&Dist[r * B][r * B], 1, block_type, 0, MPI_COMM_WORLD);
    ```
    
- Phase 2：Pivot Row / Pivot Column blocks
    處理第r行和第r列的其他區塊(扣除pivot)，透過blocks_per_process分配每個MPI process需要計算的行區間，每個process只需要處理自己分配的區塊。$blocks\_per\_process = \lceil \frac{round}{size} \rceil$。
    ```cpp
    int blocks_per_process = (round + size - 1) / size;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = start_block; i < end_block; i++) {
        if (i != r) {
            cal(B, r, i, r, 1, 1);
            cal(B, r, r, i, 1, 1);
        }
    }
    ```

- Phase 3：剩餘的blocks
    扣除i=round或是j=round的block (已經在phase1和phase2做完)
    ```cpp
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int i = start_block; i < end_block; i++) {
        for (int j = 0; j < round; j++) {
            if (i != r || j != r) {
                cal(B, r, i, j, 1, 1);
            }
        }
    }
    ```
cal函數是核心的計算，用來更新指定的block的最短路徑
```cpp
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
```
對於每個頂點對(i, j)，考慮經過中介點k是否能得到更短的路徑。k循環處理當前block的所有中介點，i是block內所有可能的起點、j用來表示終點，如此便可以更新完給定範圍內的最短路徑。

### 3-2

**資料分割**

在hw3-2採用動態的方式宣告Dist矩陣(host_D)，並且在input的時候先做了padding，讓維度成為BLOCKSIZE的整數倍，這樣可以確保所有區塊都是完整的BLOCKSIZE X BLOCKSIZE大小，避免處理最後一個不完整區塊時的記憶體對齊問題，提升coalesced memory access的效率，也省去每次檢查矩陣邊界的開銷。
整個矩陣會切分成數塊，每塊的大小為64x64，由不同的GPU block來處理，然後每塊block內部會再分割給32x32個thread來處理，每個thread負責處理這個block中的2x2區塊，這樣32x32個thread恰好能完整處理一個64x64的block。
 ```cpp
padding_n = n + (BLOCKSIZE - (n % BLOCKSIZE));

host_D = (int*) malloc(padding_n * padding_n * sizeof(int));
    
for (int i = 0; i < padding_n; ++i) {
    for (int j = 0; j < padding_n; ++j) {
        host_D[i * padding_n + j] = (i == j) ? 0 : INF;
    }
}
```

**Blocking Factor設置**

BLOCKSIZE = 64 表示block大小為64x64，TBSIZE = 32 表示每個block有32x32個thread，TILENUM = 2 表示每個thread負責處理2x2的小區塊。
因為cuda的warp size是32，使用32x32的thread配置可以最大化warp的效率，但不超過1024的限制，而TILENUM=2是因為BLOCKSIZE/TBSIZE = 2，這樣能確保每個64X64的block可以被32x32個thread完整覆蓋，最大化平行處理的能力。

架構示意圖：
![{B1C1D366-2162-4B96-B76D-F399E548B13A}](https://hackmd.io/_uploads/Byws64PSJg.png)


**CUDA block和thread配置**

演算法分成三個phase，phase1每次只需要處理一個pivot，所以只用一個block的thread；phase2需要處理和pivot同行和同列的block，使用round個block (round = padding_n / 64)；phase3處理剩下的部分，需要round x round個blocks。
```cpp=
const int round = padding_n / BLOCKSIZE;
dim3 block_dim(TBSIZE, TBSIZE);
dim3 grid_dim(round, round);

for (int r = 0; r < round; r++) {
    block_FW_phase1<<<1, block_dim>>>(dev_D, padding_n, r);
    block_FW_phase2<<<round, block_dim>>>(dev_D, padding_n, r);
    block_FW_phase3<<<grid_dim, block_dim>>>(dev_D, padding_n, r);
}
```

**不同phase的計算任務**

- Phase 1：計算當前的pivot block的距離更新
    每個thread負責處理TILENUM X TILENUM的子區塊，start_x和start_y決定了該thread負責的子區塊在block中的起點，pivot_base 是這個 block 的左上角在全域矩陣中的位置。
    
    ```cpp=
    const int start_x = threadIdx.x * TILENUM;
    const int start_y = threadIdx.y * TILENUM;
    const int pivot_base = round * BLOCKSIZE;

    ```
    然後每個thread將自己負責的Tile的資料從global memory複製到shared memory。
    
    ```cpp=
    for(int i = 0; i < TILENUM; i++) {
        for(int j = 0; j < TILENUM; j++) {
            const int global_x = pivot_base + start_x + j;
            const int global_y = pivot_base + start_y + i;
            s_D[start_y + i][start_x + j] = d_D[global_y * n + global_x];
        }
    }
    ```
    然後更新目前計算目標的最短距離，再把結果寫回global memory。
    ```cpp=
    //for loop: BLOCKSIZE * TILENUM * TILENUM
    s_D[idx_i][idx_j] = min(
        s_D[idx_i][idx_j],
        s_D[idx_i][k] + s_D[k][idx_j]
    );

    //for loop: TILENUM * TILENUM
    d_D[global_y * n + global_x] = s_D[start_y + i][start_x + j];
    ```

    
- Phase 2：更新和pivot block同行或同列的區塊
    開兩個shared memory分別記錄row和column，然後在一開始就把 pivot block 的整行和整列資料從global memory載入到shared memory(s_row 和 s_col)，以減少後續計算中對 global memory 的訪問次數，提升記憶體存取效率。
    接著每個 thread 從 global memory 中讀取自己負責的 row block 和 column block 的資料，存入 register 中 (result_row 和 result_col)，作為計算的初始值。(Register 是最快的記憶體存取層級，用於保存計算中間結果，避免多次讀取 shared memory。)。
    ```cpp=
    //for loop: TILENUM * TILENUM
    int i = threadIdx.y + bi * TBSIZE;
    int j = threadIdx.x + bj * TBSIZE;
            
    int global_i = blockIdx.x * BLOCKSIZE + i;
    int global_j = pivot_base + j;
            
    result_row[bi][bj] = d_D[global_i * n + global_j];
    s_row[i][j] = result_row[bi][bj];
            
    result_col[bi][bj] = d_D[(pivot_base + i) * n + blockIdx.x * BLOCKSIZE + j];
    s_col[i][j] = result_col[bi][bj];
    ```
    計算時，thread 使用shared memory中的 pivot block 資料 (s_row 和 s_col) 與 register 中的結果進行最短路徑更新，避免頻繁訪問 global memory 和shared memory。
    ```cpp=
    //for loop: BLOCKSIZE * TILENUM * TILENUM
    result_row[bi][bj] = min(result_row[bi][bj], 
                             s_row[i][k] + d_D[(pivot_base + k) * n + pivot_base + j]);
                
    result_col[bi][bj] = min(result_col[bi][bj],
                    d_D[(pivot_base + i) * n + pivot_base + k] + s_col[k][j]);
    ```
    整個更新完成後，將計算結果從 register 回寫到 global memory。
    ```CPP=
    //for loop: TILENUM * TILENUM
    int i = threadIdx.y + bi * TBSIZE;
    int j = threadIdx.x + bj * TBSIZE;
            
    int global_i = blockIdx.x * BLOCKSIZE + i;
    int global_j = pivot_base + j;
            
    d_D[global_i * n + global_j] = result_row[bi][bj];
    d_D[(pivot_base + i) * n + blockIdx.x * BLOCKSIZE + j] = result_col[bi][bj];
    ```
    
- Phase 3：更新剩餘未計算的區塊
    使用 blockIdx.x 和 blockIdx.y 決定當前更新的區塊，如果他們任一個等於當前的round表示該區塊在前面的phase1和phase2已經計算過，就可以直接跳過。然後一樣從 global memory 中將 pivot block 的整行和整列資料載入到共享記憶體 (s_row 和 s_col)，threads再從 global memory 中讀取自己負責的目標 block 資料並存入 register (s_remaining)，作為初始值。基本上和phase2邏輯一樣，只是計算的index不同。
    
### 3-3
None.

## Profiling Results (hw3-2)
以testcase: c20.1來當作測試資料，並使用nvprof獲取所有kernel的執行資料
![{D0AC5968-76AE-46D0-A39B-6A22F571EA54}](https://hackmd.io/_uploads/SkBs7hWBJx.png)

根據結果，block_FW_phase3是執行時間最長的kernel，再針對該kernel使用--metrics 收集詳細效能數據：
![{DDA1FA37-6D22-49AE-98A8-8060026B442E}](https://hackmd.io/_uploads/Bkm_Eh-rJx.png)


## Experiment & Analysis
a. System Spec: apllo-gpu

b. Blocking Factor (hw3-2)
測資：c20.1
由於當BLOCKSIZE=128時會超過shared memory的限制而出現報錯，所以實驗只測試8, 16, 32, 64

<img src="https://hackmd.io/_uploads/BkPmDaWrJe.png" width="45%"> <img src="https://hackmd.io/_uploads/SksHv6ZB1g.png" width="45%">

**Computation Performance**

當Blocking Factor增加時，整體效能有顯著的提升，可能的原因
- 因為更大的Blocking Factor允許更多的計算工作在每個block內完成，因此減少了存取記憶體的頻率。
- 更大個block size表示更多資料能暫存在shared memory中，減少了對global memory的存取。
- Blocking Factor變大後，每個thread block的記憶體訪問模式可能更加coalesced，而提高了access的效率。

不過從8到16增加了3.5倍，16到32增加約3.2倍，32到64增加約1.2倍，顯示效能成長並非線性的，有可能是因為受限於GPU硬體資源的限制(如shared memory的大小或記憶體頻寬等)。

**Memory Performance**

當Blocking Factor較小時，更依賴global memory，因此性能較低；當Blocking Factor較大時，shared memory性能顯著提升，因為當blocking size變大後，每個thread block能處理更多資料，而能高效的利用GPU的shared memory。
對於Floyd-Warshall這種需要頻繁訪問資料的演算法，利用shared memory可以大幅減少global memory的訪問次數，加上受益於shared memory的高頻寬，因而導致了顯著的性能成長。(可參考實驗c. optimization)

c. Optimization (hw3-2)
測資：p12k1
![image](https://hackmd.io/_uploads/ByDKNgfSyl.png)

CPU版本是hw3-1的實作結果，可以看到即使是一個簡單的GPU程式(未加入優化技巧)就能贏過他。而加入記憶體對齊的優化後，有效的提高了memory access的效率而提高了性能。
其中shared memory的優化顯示了最顯著的性能提升，如先前實驗所分析的，Floyd-Warshall是記憶體密集型的演算法，頻繁的memory acess是他的瓶頸。
最後的Occupancy Optimization和Blocking Factor Tuning則提升了資源的利用率，並在計算和記憶體訪問需求中取得平衡，來達到更佳的性能表現。

d. Weak scalability (hw3-3)
None

e. Time Distribution (hw3-2)
![image](https://hackmd.io/_uploads/Bkp5WWfHyg.png)


| Test Case | N | Compute time (ms) | Memory copy time (ms) | I/O time (ms)
| --------  | -------- | --------- |--------- | --------- |
| c16.1     | 1500     | 4.298     | 3.948 | 59 |
| c20.1     | 5000     | 131.195   | 45.122 | 390 |
| p11k1     | 11000    | 11930.137 | 219.11    | 882 |
| p12k1     | 11817    | 14647.54  | 250.86     | 1202 |

隨著N變大，I/O time和Computing time都有大幅的增加，尤其是computing time影響最大，所以我們用GPU平行化了計算後，可以使得整體的runtime加速許多。

## Experiment on AMD GPU
a. hipify-clang hw3-2.cu -> make hw3-2-amd -> ./hw3-2-amd

b. Compare the difference between Nvidia GPU and AMD GPU.

| Feature | Nvidia GPU | AMD GPU |
| ------- | ---------- | ------- |
|架構      | SM (warp-based) | CU (SIMD) |
|記憶體     | GDDR, unified memory | HBM, ROCm-managed memory |
|效能       |compute/memory balance | High memory bandwidth |

Nvidia GPU使用streaming multiprocessors (SMs)作為主要的計算單元，SM是基於warp (32 threads)的執行進行最佳化，並且能有效處理branch控制；而AMD GPU使用Compute Units (CUs)作為主要計算單元，通常每個單元擁有更多核心，能提供更高的peak throughput。
在記憶體方面Nvida GPU通常使用GDDR記憶體，而AMD GPU則使用HBM (高頻寬記憶體)，提供更高的記憶體頻寬。

c. Share your insight and reflection on the AMD GPU experiment.
![image](https://hackmd.io/_uploads/SJhECYfSJl.png)

從實驗結果我們可以看出對於N較小(小規模)的矩陣，兩者的性能相當，而到了較大的N時，我們就可以明顯的看出AMD GPU的優勢了。因為Floyd-Warshall 演算法需要頻繁訪問global memory來更新距離矩陣，而AMD 的HBM提供了顯著更高的頻寬，使其在中大型矩陣的任務上表現優異。

