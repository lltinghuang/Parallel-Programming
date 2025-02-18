# Parallel Programming HW1

## Implementation
1. How do you handle an arbitrary number of input items and processes?
先計算每個process需要處理的資料量: num = n / num of process，如果n不能被整除，就讓前幾個process每個多處理一個item，這樣可以讓工作更平均的分散給processes(彼此最多只會差一個input item)，所以在切分的時候大小就直接取ceiling()。

2. How do you sort in your program?
一開始分成size個process，然後每個process各自通過MPI /O來讀取資料，並且分配記憶體(data, buffer, tmp)，也預先把自己內部的資料排好(local sort)，最後再應用Odd-Even Sort演算法來完成這些process的溝通、排序(global sort)。 
![image](https://hackmd.io/_uploads/Byg9AWhlJl.png)
Odd-Event Sort 會分成Even Phase & Odd Phase，Even Phase 時rank為偶數的process會跟rank為奇數的process比較，也就是Even phase的時候，偶數rank的process會去看他右邊那個鄰居process的東西，而因為我們已經先做好local sort了，所以此時可以先比當前process的最後一個元素(maximum)和右鄰居的第一個元素(minimum)來判定是否完成排序，如果當前process的最大值小於右鄰居的最小值，表示這兩個process已經排好，否則兩個process之間就需要進行溝通、交換資訊來重新排序；Odd phase也是相同的概念。
而最差情況就是最大的元素在 P0，最小的在 P(size-1)，在每一輪，資料都會向正確的方向移動至少一個位置，最遠的資料需要移動 size-1 個位置，因此我們最多花size次就能完成排序。

3. Other efforts you’ve made in your program.
使用`boost::sort::spreadsort::float_sort`來對 local array 進行排序，他是Boost函式庫中一個特殊的排序演算法，專門針對float進行優化，對於float通常比std::sort快，適合處理大量的float數據。
我本來將某些計算寫成了一些function，但是把他們移除、將內容併入main()之後，成功提升了performance。我想這是因為function call overhead所導致的，改寫後減少了一些stack frame和傳遞參數、return address處理的成本。對於重用性較高的function，則採取了折衷的方案，使用`inline`來提示編譯器最佳化。

## Experiment & Analysis
### 1. Methodology
##### (a). System Spec
課堂提供的Apollo平台
##### (b). Performance Metrics
- Preprocess time 每個Process 讀取data後的預先排序和分配/釋放記憶體空間的時間 
- IO time = MPI_File_open + MPI_File_close + MPI_File_read_at + MPI_File_write_at 讀取和寫入各個Process 自己資料的時間
- Communication time = MPI_Isend + MPI_Irecv + MPI_Wait 各個程式之間的溝通資料所花費的時間
- Computation time = Total time - (Preprocess time + IO time + Communicatin time) 處理自己和相鄰process的data以及其餘判斷所需的計算

一開始我是用Nsight System去得到這些數據，但是因為我的Nsight System有些異常而導致某些功能不能用，所以最後是用MPI_Wtime()來獲取這些資料。
### 2. Plots: Speedup Factor & Profile
#### Experimental Method: 
我挑的測資是`35.in`，`n: 536869888`
#### Analysis of Results & Discussing:
##### Time profile
<div style="display: flex; justify-content: space-around;">
    <img src="https://hackmd.io/_uploads/r1V1Yehxyx.png" width="45%">
    <img src="https://hackmd.io/_uploads/HyVltlngyl.png" width="45%">
</div>

當只有一個process的時候，我們可以清楚看出Preprocess的部分是效能瓶頸，在有多個process時，每個process分到比較少的資料，因此在前置處理和內部排序的負擔都降低了，可見隨著平行度提升，在preprocess的地方有較明顯的改善。
而在有較多process的時候，可以發現communication time上升了，這是符合我們預期的，在多個process的時候，各別的計算量應該會下降，但相對的會提升溝通成本。
此外，比較一個node跟三個node的結果也可以發現，node間溝通沒有使communication time 有明顯的增加，造成瓶頸的原因可能還是因為資料變零碎造成交換的次數變多。 
從實驗結果可以看到，雖然CPU time在單一node的時候是主要的效能瓶頸，但隨著平行度增加，CPU time (Preprocess + computation)有顯著的下降，而IO time的部分，雖然讀寫的資料量變少卻沒有甚麼影響的感覺，我想是因為資料讀寫一定需要file open/close 的動作，這些file access的行為具有固定的開銷，所以我認為整體加速的瓶頸在於IO time。
其中一個有實現的優化是連續開啟檔案，我原本是打開input_file然後做完所有操作，直到最後要寫入時才打開output_file，但我改成一開始就先把兩個檔案都打開，就有比本來的做法快3s(based on scoreboard)，我想這可能是因為系統會預先讀取鄰近的file system資料或是受益於OS的快取機制以及I/O scheduling的緣故。
另外或許可以嘗試做批次讀寫跟使用其他專門的I/O library。

##### Speedup
<div style="display: flex; justify-content: space-around; margin-top: 10px;">
    <img src="https://hackmd.io/_uploads/rkkXtghx1l.png" width="45%">
    <img src="https://hackmd.io/_uploads/SyUmtghgJl.png" width="45%">
</div>

實驗結果可以看出最終的speedup factor大約都到3左右就上不去了，大約到4個process之後就沒有明顯的進步。
我想這可能和Odd-Even Sort這個演算法的特性有關，從time profile可以看出computation time沒有因為數據變少而有太多的降低，而頻繁的溝通或許也在某種程度上抵銷了平行化所帶來的優勢，所以導致加速結果不理想。

##### Optimization Strategies:
![image](https://hackmd.io/_uploads/rkmybWhe1e.png)
![image](https://hackmd.io/_uploads/r1Wx-Wne1g.png)

在原先的做法中本來有early stopping的機制，但因為上課時老師有提到branch會讓效能變差，為了達到early stopping也必須先做Allreduce來檢查是否完成所有的排序，增加了溝通的成本，所以一個優化方法就是拿掉這個early stoping的機制，讓他做fully iteration。
實驗結果有略為減少一點時間，特別是在communication time上獲得了進步，但這也增加了computation的負擔，所以就最後的結果來說並沒有顯著的優化成效。
## Experiences / Conclusion
這是我第一次寫平行程式，一開始我覺得最困難的地方是要轉換自己的思維，把sequential執行的計算，想辦法拆分讓好幾個process可以一起做。但後面在進行實驗想辦法優化、加速的時候，才發現比起對單一process的任務做優化，重點可能還是在於整個平行程式的設計，而其中涉及一些演算法和計算機結構、作業系統的知識，如何整合這些技術以及活用，這才是真正困難所在的地方。