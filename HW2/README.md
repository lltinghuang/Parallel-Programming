# Parallel Programming HW2

## Implementation
### How you implement each of requested versions, especially for the hybrid parallelism.
#### Pthread
- 用core的數量來當作thread的數量。
- 使用一個共享的變數cur_row來記錄下一個要計算的row，並使用atomic操作來處理同步化的問題。
- 在每個thread執行的function內有一個while loop，在當他計算完當前這行後就會繼續去拿下一行來計算，當cur_row大於圖片的height表示所有行都計算完了，就會結束整個計算。
- 最後主程式再寫入整個結果、輸出圖片。
#### Hybrid
- 首先，我使用靜態的方式分配需要計算的row給各個MPI process，

> 例如4個process: 

> Process 0: row 0, 4, 8, 12...

> Process 1: row 1, 5, 9, 13... 

> Process 2: row 2, 6, 10, 14... 

> Process 3: row 3, 7, 11, 15...

- 再用OpenMP的dynamic scheduling，讓每個thread在完成當前行的計算後自動地去抓下一行來計算(類似Pthread版本的作法)，最後使用MPI_Gather來收集所有process的結果。
- 不過MPI_Gather收集到的數據並不是按最終完整圖像的順序存的，而是按照各個process分段儲存的，是依據rank的順序拼接，因此要write_png中重新排序

    `int p = buffer[((tmp % size) * row_seg + tmp / size) * width + x];`

--  `tmp = height - 1 -y`：將當前處理的行進行翻轉後的index，使得在 buffer 中從下往上對應。確保繪製的時候從buffer的底部(圖像頂部)開始

--  `tmp%size`：找到該行是由哪個process (rank)處理的

--  `row_seg`：是每個proces分配到的行數

--  `tmp/size`：計算該行在process的分段中的第幾段

--  `(tmp % size) * row_seg + tmp / size`：來得到該行在整個buffer(收集到的數據)中的正確位置

-- ` *width + x`：乘以 width 可以找到該行的起始pixel，再加上 x 找到具體pixel的位置

### How do you partition the task?
#### Pthread
- 以行為基本單位
- 用atomic動態分配每一行，每個thread完成當前處理的行後會透過get_pos()來取得新的行，直到所有行都處理完成

#### Hybrid
- 靜態分配：把row平均分配給不同的MPI process (跳rank分，而非連續的row分配給同一個rank)
- 動態分配：在每個MPI process內部使用OpenMP做dynamic scheduling，每個thread在完成當前的行處理後，會自動從剩餘未處理的行中取得新的工作
```
 #pragma omp parallel num_threads(ncpus)
    {
        #pragma omp for schedule(dynamic)
        for(int row = rank; row < height; row += size) {
            process_row(row, image);
        }
    }
```

### What technique do you use to reduce execution time and increase scalebility?
#### Pthread
- SIMD vectorization (使用AVX-512)：每行一次處理32個像素（四組，每組8個，64(double)*8 = 512）
- 使用原子計數器 `std::atomic<int> cur_row` 動態分配工作，避免靜態分配可能造成的負載不均衡。(經實作發現使用atomic比mutex快)
- 預先計算每列的 y0 值，減少重複計算
- aligned allocation
使用 aligned_alloc 函數確保 image 和 y0_values 的記憶體分配滿足快取行對齊需求，可以減少cache miss的情況，而提高計算性能
- 將thread綁定到特定 CPU 核心，提高cache的命中率，減少核心間的任務切換開銷
- 用mask判斷來減少branch prediction和記憶體存取
```
// 使用 bool array - 多個分支判斷
if(!active1[0] && !active1[1] && !active1[2] && !active1[3] && 
   !active2[0] && !active2[1] && !active2[2] && !active2[3]) 
    break;

// 使用 mask - 單一分支判斷
if(!active_mask) break;
```
#### Hybrid
- 靜態循環分配MPI要處理的行，使得每個MPI可以獨立計算自己要處理的行，不需要和其他MPI process溝通或同步，最小化了溝通成本，只需要最後一次MPI_Gather收集結果
- MPI process內部做Dynamic Load Balancing（使用`#pragma omp for schedule(dynamic)`），避免負載不平衡並提高CPU利用率
- SIMD vectorization (使用AVX-512)：每行一次處理32個像素（四組，每組8個，64(double)*8 = 512）
- aligned allocation
- 用mask判斷
### Other efforts you made in your program
過程中我曾遇到一次judge所有測資會有部分測資沒過，但是單獨測是那些測資時卻又通過的問題，參考了討論區同學的意見："某些特殊的平行條件下可能會觸發 segment fault，或是大部分時間會有違規的 memory access"，而我主要的設計核心在Pthread和hybrid是相似的，所以我想可能是同步化的地方出現了問題。
```
//new
int get_pos() {
    int row = cur_row.fetch_add(1);
    return (row < height) ? row : -1;
}

//original
int get_pos() {
    int row = cur_row.load();
    if(row >= height) {
        return -1;
    }
    return cur_row.fetch_add(1);
}
```
經過分析研究後，我發現這個地方可能有race condition，在執行load() 和 fetch_add(1) 之間的間隙中，可能其他thread已經對 cur_row + 1。
新版本則將 load() 和 fetch_add(1) 合併為一個原子操作 fetch_add(1)，保證了讀取和增加的操作在一個原子步驟中完成，這樣其他thread就無法在這兩個操作之間插入了。

## Experiment & Analysis
### Methodology
a. System Spec
- 課堂提供的server

b. Performance Metrics
- For Pthread: 用`std::chrono::high_resolution_clock`
- For Hybrid: 用`MPI_Wtime`


---

* Preprocess time：記憶體分配和預計算
* Compute time：主要的 Mandelbrot 計算
* I/O time：write PNG 
* Communication time：在hybrid版本中，把所有MPI process結果蒐集回來的時間

### Plots:Scalability & Load Balancing & Profile
#### Experimental Method:
- Test case: strict34.txt

#### Experiment 1 - Pthread
- Parallel Configuration:
    - 1 node 
    - 1 process
    - 1, 3, ..., 19 threads
<div style="display: flex; justify-content: space-around; margin-top: 10px;">
    <img src="https://hackmd.io/_uploads/S1oNaAX-kx.png" width="45%"> <img src="https://hackmd.io/_uploads/Bk7Zb9V-1g.png" width="45%">
</div>    
    從實驗結果可以看到隨著thread數量增加，執行時間有大幅的減少，其中最主要影響的是computation time的部分(bottleneck)，也因此在speedup的部分也呈現的線性的成長。不過離ideal還是有一點差距，尤其在thread數量到11之後明顯變得趨緩，推測可能的原因是多個thread同時寫入image的不同位置造成的記憶體頻寬競爭，以及需要競爭原子變數cur_row來獲取下一行，而導致的結果。

#### Experiment 2 - Hybrid
- Parallel Configuration:
    - 1 node
    - 1 process
    - 1, 3, ..., 19 threads
<div style="display: flex; justify-content: space-around; margin-top: 10px;">
    <img src="https://hackmd.io/_uploads/B1rl00mbJx.png" width="45%"> <img src="https://hackmd.io/_uploads/BJSw0CXW1g.png" width="45%">
</div>        

- Parallel Configuration:
    - 1 node
    - 3 process
    - 1, 3, ..., 19 threads
<div style="display: flex; justify-content: space-around; margin-top: 10px;">
    <img src="https://hackmd.io/_uploads/rJWaRAmZJl.png" width="45%"> <img src="https://hackmd.io/_uploads/r1NRACmZ1x.png" width="45%">
</div>                                                    從實驗結果可以看到隨著thread數量增加，執行時間有大幅的減少，因為我設計Hybrid版本的邏輯和Pthread版本相似，所以我想這樣的結果是合理的。不過在process=3的speedup factor上顯然偏離ideal很多，我想這是因為和process=1的時候相比，在相同thread的狀況下computation time下降了2~3倍，但維持了相同的I/O time，而導致增加thread所帶來的加速效果變得沒有那麼顯著，或許這部分可以將wring png的工作平行化來達到加速。
                                                       
#### Experiment3 - Load balancing
- Test case: strict34.txt & slow12.txt
- Parallel configuration:
    - 2, 4, ..., 10 threads (for Pthread)
    - 2, 4, ..., 10 processes & each has 10 threads (for Hybrid)
<div style="display: flex; justify-content: space-around; margin-top: 10px;">
<img src="https://hackmd.io/_uploads/BydLJJ4Z1l.png" width="45%"> <img src="https://hackmd.io/_uploads/ryDP11EZ1g.png" width="45%">
</div>

因為測量出來每個thread執行的時間看起來都很接近，所以我使用了一個負載平衡的指標來繪圖分析
$$
\text{balance} = 1 - (\text{max\_diff} - \text{average})*100 \%
$$

從結果我們可以看出Pthread版本在兩個不同測資的表現上都維持相當高的負載平衡度（99.5%以上），顯示了用原子變數來進行動態工作分配的策略效果很好。

而Hybrid版本在較少process的時候也維持很好的負載平衡度，但當process增加到10的時候，不論對於哪個測資，負載平衡度都下降很多，其可能原因是靜態分配的限制(row+=size)，可能某些MPI process會分到計算密集的區塊，當proces較多的時候，這個不平衡的問題越容易發生；另一方面，多層次的平行化(MPI+OpenMP)造成資源競爭，以及MPI process之間的溝通成本增加，所以導致這樣的結果。

### Discussion
#### Compare and discuss the scalability of your implementations
根據上面的實驗結果，可以看到這Pthread和Hybrid版本其實都scale得不錯，我想是因為Mandelbrot set的特性可以很好的被平行化來運算，而且大部分是可以獨立計算的，因此可以減少掉很多溝通成本。
#### Compare and discuss the load balance of your implementations
從Experiment 3可以看出來在Hybrid的process小於10的狀況下，Pthread版本和Hybrid版本都有相當好的負載平衡度，但Hybrid版本在使用很多process的狀況下，MPI process之間的分配就有較大的可能會負載失衡。
## Experience & Conclusion
在這次作業中主要的優化都是在研究如何做load balancing以及透過vectorization來加速，感覺有比第一次作業時更加的進步，比較有概念優化要往哪個方向去做，並且分析可能的瓶頸或是出bug的地方，然後進行修正。從speedup factor的結果也可以看到，程式有更高的平行度了，就還蠻有成就感的。

不過這次有遇到一些比較神奇的bug，像是在使用avx512的時候，compile用O1會過、用O3就不會過了，這似乎是不同優化程度會讓avx512有不同的表現，而這是我過去寫程式比較不會去思考到的問題。