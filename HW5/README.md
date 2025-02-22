# PP HW 5 Report 

> - Please include both brief and detailed answers.
> - The report should be based on the UCX code.
> - Describe the code using the 'permalink' from [GitHub repository](https://github.com/NTHU-LSALAB/UCX-lsalab).

## 1. Overview
> In conjunction with the UCP architecture mentioned in the lecture, please read [ucp_hello_world.c](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/pp2024/examples/ucp_hello_world.c)
1. Identify how UCP Objects (`ucp_context`, `ucp_worker`, `ucp_ep`) interact through the API, including at least the following functions:
    - `ucp_init`：creates and initializes a "UCP application context", and must be called before any other UCP function call in the application. It takes 3 parameters:
        - config: UCP configuration descriptor allocated through "ucp_config_read()" routine.
        - params: User defined ucp_params_t configurations for the "UCP application context".
        - context_p: Initialize "UCP application context".
    - `ucp_worker_create`: create a worker object. This routine allocates and initializes a ucp_worker object. Each worker is associated with one and only one ucp_application context. It takes 3 parameters:
        - context: Handle to "UCP application context".
        - params: User defined ucp_worker_params_t configurations for the "UCP worker".
        - worker_p: A pointer to the worker object allocated by the UCP library.
    - `ucp_ep_create`:create and connect an endpoint on a "local worker" for a destination "address" that identifies the remote "worker". It takes 3 parameters:
        - worker: Handle to the worker; the endpoint is associated with the worker.
        - params: User defined ucp_ep_params_t configurations for the "UCP endpoint".
        - ep_p: A handle to the created endpoint.
        
    首先，利用`ucp_config_read`讀取環境變數以及設定有關ucp context的參數，然後呼叫`ucp_init(&ucp_params, config, &ucp_context)`，他會將用戶需求(ucp_params)和UCX配置(根據環境變數或是default值)結合，來創建一個`ucp context`，這個context是UCX所有操作的基礎。
    接著，用`ucp_worker_create`創建`ucp_worker`，它是UCX通信操作的kernel object，負責管理通信進度、執行消息的收發以及和底層硬體的互動溝通。通過設定UCP_WORKER_PARAM_FIELD_THREAD_MODE in work_params，可以確認在ucp_worker_create 的時候使用的是single thread的方式，以避免不必要的lock操作，然後再透過`ucp_worker_query`函數搭配`ucp_worker_attr_t`的field mask (UCP_WORKER_ATTR_FIELD_ADDRESS)就能查詢worker的地址資訊，並將其先存至變數中來給後續的通信使用。
    創建好worker並獲取地址之後，server和client通過Out-of-Band的方式交換地址資訊並建立連接。
    
    **Client Side**
    Client通過`getaddrinfo`獲取Server的地址資訊，並嘗試連接所有可用的地址。成功後，使用Socket與Server交換地址資訊。首先，Client接收Server發送的地址長度 (`peer_addr_len`)，然後接收具體的地址內容 (`peer_addr`)，最後進入client side想要執行的`run_ucx_client`：
    
    - 首先，將 Endpoint 相關的設定存入 ep_params，包括 ep_addr（也就是 server 的地址）。接著執行 ucp_ep_create，建立對應 server side 的 Endpoint。
        
    - 準備一個 message buffer，並將 client side 的地址包含在內，通過 ucp_tag_send_nbx 傳送。這是一個非阻塞的發送函數，會立即返回。因此，需要一個等待函數來檢查消息是否已成功發送。
        
    - ucx_wait 函數負責這個目的。該函數包含一個 while 迴圈，不斷執行 ucp_worker_progress，直到條件 !request->completed 被滿足。ucp_worker_progress 會顯式推進 Worker 上所有通信操作的進度，並通過阻塞例程推動通信狀態的進展。在這個情況下，它等待 ucp_tag_send_nbx 的操作完成。
        
    - 接下來，等待接收來自 server 的測試字符串。這個過程需要一個 for 迴圈，不停地調用 ucp_tag_probe_nb，檢查是否有與當前 tag 相符的 message。如果有，就結束迴圈；如果沒有，則繼續調用ucp_worker_progress，處理未完成的通信操作。如果沒有任何未完成的通信，則可以讓 CPU 進入休眠狀態。通過 ucp_worker_wait等待事件發生。
        
    - 在確認收到一個 message 之後，可以通過 ucp_tag_msg_recv_nbx 非阻塞地接收該 message。搭配 ucx_wait，可以像之前發送消息時一樣，等待接收操作的完成。在設定接收函數的參數時，使用 ucp_request_param_t 設定回調函數以及要傳入的 datatype。如果一切順利，最後可以釋放過程中使用的 socket 連接以及 message memory 等資源。
        
    **Server Side**
    Server使用`getaddrinfo`獲取適合的Socket描述符，並通過`bind`函數綁定到指定的服務。成功連接Client後，Server將自己的地址長度 (`local_addr_len`) 和地址內容 (`local_addr`) 發送給 Client，為通信建立基礎。進入`run_ucx_server`：
    - 持續執行 `ucp_worker_progress` 和 `ucp_tag_probe_nb`，直到`msg_tag`不為NULL。然後根據 `info_tag` 提供的長度為該消息分配記憶體。接著使用 `ucp_tag_msg_recv_nbx` 執行接收過程，並利用 `ucx_wait` 確認接收是否完成。在將收到的 `msg` 填入 client 的地址後，server 開始向 client 傳送測試字符串。

    - 根據 `peer_addr`，server 創建一個 endpoint，並使用 `ucp_ep_create` 與遠端目標建立連接。

    - 在設置好 endpoint 的連接後，server 使用 `ucp_tag_send_nbx` 傳送測試消息，並通過 `ucx_wait` 等待操作完成。當退出 `ucx_wait` 的迴圈時，表示消息已經發送。如果 `ucx_wait` 的返回狀態是 `UCS_OK`，說明通信成功。最後一步是使用 `flush_ep` 刷新該 endpoint。

    - 在 `flush_ep` 函數內部，包含了 `ucp_ep_flush_nbx`。這個函數會刷新 endpoint 上所有未完成的 AMO 和 RMA 通信。在此呼叫之前發出的所有 AMO 和 RMA 操作，當函數返回時，會保證已在source端和destnation端完成。在 `ucp_hello_world` 中，採用阻塞方式實現這一刷新過程。一邊通過 `ucp_worker_progress` 處理 worker 上其他 endpoint 的通信，一邊等待當前 endpoint 的刷新操作結束。
    
2. UCX abstracts communication into three layers as below. Please provide a diagram illustrating the architectural design of UCX.
    - `ucp_context`: 定義在`ucp_context.h`裡面的一種結構，負責管理與特定UCP instance相關的資源，包含UCT components、memory domain resources，並且記錄application的memory registration情況。此外，還透過feature記錄支持的通信功能，像是`tag`通信，以及是否支持wake-up。
    - `ucp_worker`: 定義在`ucp_workder.h`裡面的一種結構，包含了`ucp_context handler`、woker host name和一個用來統計endpoint數量和狀態的結構。通常一個thread對應一個worker。worker負責處理與傳輸相關的操作，像是interrupts、建立連接以及完成事件等等。此外，還記錄worker的thread mode、事件、名稱以及客戶端ID的資訊。

    - `ucp_ep`: 定義在`ucp_ep`裡面的一種結構，為UCX協議層的endpoint，表示和遠端worker的連接，負責維護跟遠端的連接資源、管理通信路徑(lanes)並抽象底層傳輸層(如TCP)。
    
> Please provide detailed example information in the diagram corresponding to the execution of the command `srun -N 2 ./send_recv.out` or `mpiucx --host HostA:1,HostB:1 ./send_recv.out`
    
![{63EC4141-808C-47EA-A87C-73EE6DD8ADA2}](https://hackmd.io/_uploads/BkgBarYSJl.png)

![{B90F8ED2-F942-4874-AE0E-C6E39D5E8A2A}](https://hackmd.io/_uploads/Hk10MtFr1l.png)



3. Based on the description in HW5, where do you think the following information is loaded/created?
    - `UCX_TLS`：
        UCX_TLS可能在解析`ucp_context`的配置過程中被load進來，所以應該存放在`ucp_context.c`或跟它相關的地方。
    - TLS selected by UCX：
        我認為這發生在`ucp_ep_create`的過程之中，它的流程如下：`ucp_ep_create` -> `ucp_wireup_init_lanes` -> `ucp_wireup_select_lanes` -> `ucp_wireup_search_lanes`。
因為 search_lanes 函數會從最快的protocol開始測試是否可以連接成功，如果可以，就選擇這個協議。這個過程可以根據協議的分數選出最佳的傳輸方式 (best transports)。

## 2. Implementation
> Please complete the implementation according to the [spec](https://docs.google.com/document/d/1fmm0TFpLxbDP7neNcbLDn8nhZpqUBi9NGRzWjgxZaPE/edit?usp=sharing)
> Describe how you implemented the two special features of HW5.
1. Which files did you modify, and where did you choose to print Line 1 and Line 2?
    修改了`ucp/core/ucp_worker.c`, `ucs/config/parser.c`和`ucs/config/type.h`。
    - `ucp_worker.c`:
    根據`mpiucx -x UCX_LOG_LEVEL=info`的資訊，可以找到在哪些地方印出了所需的UCX_TLS和這個protocol用的key字串，也就發現`ucp_worker_print_used_tls`會輸出每個傳輸層的所有資訊，Line 2的內容要在這裡被輸出。
    在`ucp_worker_print_used_tls`這個function加入：
    ```cpp=
    ucp_config_print(NULL, stdout, NULL, UCS_CONFIG_PRINT_TLS); //輸出UCX配置中與傳輸層協議相關的資訊
    fprintf(stdout, "%s\n", ucs_string_buffer_cstr(&strb));
    ```
    
    - `parser.c`:
    在`ucp_config_print()`這個function裡面加入迴圈來遍歷環境變數，找出`UCX_TLS`然後把它印出來。
    ```cpp=
    // TODO: PP-HW-UCX
        if (flags & UCS_CONFIG_PRINT_TLS) {
            char **e;
            for (e = environ; *e != NULL; e++) {
                if (strncmp(*e, "UCX_TLS", 7) == 0) {
                    fprintf(stream, "%s\n", *e);
                }
            }
        }
    ```
    
    `type.h`: 加入新的flag來處理TLS
    ```cpp=
    /**
     * Configuration printing flags
     */
    typedef enum {
        UCS_CONFIG_PRINT_CONFIG          = UCS_BIT(0),
        UCS_CONFIG_PRINT_HEADER          = UCS_BIT(1),
        UCS_CONFIG_PRINT_DOC             = UCS_BIT(2),
        UCS_CONFIG_PRINT_HIDDEN          = UCS_BIT(3),
        UCS_CONFIG_PRINT_COMMENT_DEFAULT = UCS_BIT(4),
        UCS_CONFIG_PRINT_TLS             = UCS_BIT(5) //added
    } ucs_config_print_flags_t;
    ```
    
2. How do the functions in these files call each other? Why is it designed this way?
    `ucp_worker_print_used_tls()`負責列出使用中的TLS(Transport Layers)，並印出line 2 (因為它會收集跟worker layer相關的TLS資訊)。而在這個函數內，會調用`ucp_config_print()`，然後這個函數又會呼叫`ucs_config_parser_opts()`來印出line 1。
    - Line 2: `ucp_ep_create` -> `ucp_ep_create_to_sock_addr` -> `ucp_ep_init_create_wireup` -> `ucp_worker_get_ep_config` -> `ucp_worker_print_used_tls` -> `printf`
    - Line 1: `ucp_worker_print_used_tls` -> `ucp_config_print` -> `ucs_config_parser_print_opts` -> `ucs_config_parser_print_env_vars`
    
    由於 Context 是 UCX 框架的核心，它需要統一管理所有的配置，因此 `ucp_config_print() `被定義在` ucp_context.c` 中，用於打印和處理與 Context 相關的配置。而 UCS 提供了如 `ucs_config_parser_print_opts `的服務函數，專門處理底層的配置解析，這些函數能夠被 UCP 和 UCT 等多個模組重用，這樣的分層設計能夠升程式的模組化與抽象性，也方便未來的擴展與維護。
    
3. Observe when Line 1 and 2 are printed during the call of which UCP API?
    在`ucp_ep_create()`被調用的時候
4. Does it match your expectations for questions **1-3**? Why?
    跟UCX_TLS 的部分不相符，但與 TLS 的選擇部分一致。這是因為 UCX 程式將從全局環境中收集的配置與 `ucp_context` 的功能分開處理。儘管某些配置是用於 `ucp_context` 的，但它們會被放入 UCS 中，作為整個 UCX 系統的服務部分。

5. In implementing the features, we see variables like lanes, tl_rsc, tl_name, tl_device, bitmap, iface, etc., used to store different Layer's protocol information. Please explain what information each of them stores.
    - `lanes`: 定義在`ucp/core/ucp_ep.h`，每個lane通常對應一個具體的通信資源，而且單個endpoint可以有多個lane。內容定義如下，儲存了resource index、目標端memory domain的index、目標系統設備、設備路徑的index、這個lane的操作類型以及對端能接收的最大fragment大小。
        ```cpp=
        typedef struct ucp_ep_config_key_lane {
            ucp_rsc_index_t      rsc_index; /* Resource index */
            ucp_md_index_t       dst_md_index; /* Destination memory domain index */
            ucs_sys_device_t     dst_sys_dev; /* Destination system device */
            uint8_t              path_index; /* Device path index */
            ucp_lane_type_mask_t lane_types; /* Which types of operations this lane
                                                was selected for */
            size_t               seg_size; /* Maximal fragment size which can be
                                              received by the peer */
        } ucp_ep_config_key_lane_t;
        ```

    - `tl_rsc`: 是`uct_tl_resource_desc_t`的一個instance，用於描述網路資源 (參`ucp/core/ucp_context.h`),包含關於傳輸層資源的詳細資訊，如傳輸層的名稱、設備名稱、設備類型和系統設備。
        ```cpp=
        typedef struct uct_tl_resource_desc {
            char                     tl_name[UCT_TL_NAME_MAX];   /**< Transport name */
            char                     dev_name[UCT_DEVICE_NAME_MAX]; /**< Hardware device name */
            uct_device_type_t        dev_type;     /**< The device represented by this resource
                                                        (e.g. UCT_DEVICE_TYPE_NET for a network interface) */
            ucs_sys_device_t         sys_device;   /**< The identifier associated with the device
                                                        bus_id as captured in ucs_sys_bus_id_t struct */
        } uct_tl_resource_desc_t;
        ```
    - `tl_name`: `uct_tl_resource_desc`的一個屬性，用於描述傳輸層的名稱(像是`ud_verbs`)。
    - `tl_device`: 類型為`uct_tl_device_resource_t` (參`uct/base/uct_iface.h`)，負責描述與傳輸設備相關的資訊，包括硬體設備名稱跟類型。 
       ```cpp=
         /**
         * Internal resource descriptor of a transport device
         */
        typedef struct uct_tl_device_resource {
            char                     name[UCT_DEVICE_NAME_MAX]; /**< Hardware device name */
            uct_device_type_t        type;       /**< The device represented by this resource
                                                      (e.g. UCT_DEVICE_TYPE_NET for a network interface) */
            ucs_sys_device_t         sys_device; /**< The identifier associated with the device
                                                      bus_id as captured in ucs_sys_bus_id_t struct */
        } uct_tl_device_resource_t;
        ```
    - `bitmap`: 每一個bit對應一個傳輸層資源，設為1表示正在被使用，可以用於追蹤UCX系統中的傳輸層資源使用情況，進行高效的管理，並幫助worker選擇最佳的接口(iface)。
       ```cpp=
         /**
         * UCP TL bitmap
         *
         * Bitmap type for representing which TL resources are in use.
         */
        typedef ucs_bitmap_t(UCP_MAX_RESOURCES) ucp_tl_bitmap_t;
        ```
    - `iface`: UCT的通信介面，包含所有傳輸接口可能的操作(`uct_iface_ops_t`)，定義在`uct/api/tl.h`。
       ```cpp=
         /**
         * Communication interface context
         */
        typedef struct uct_iface {
            uct_iface_ops_t          ops;
        } uct_iface_t;
        ```

## 3. Optimize System 
1. Below are the current configurations for OpenMPI and UCX in the system. Based on your learning, what methods can you use to optimize single-node performance by setting UCX environment variables?

```
-------------------------------------------------------------------
/opt/modulefiles/openmpi/ucx-pp:

module-whatis   {OpenMPI 4.1.6}
conflict        mpi
module          load ucx/1.15.0
prepend-path    PATH /opt/openmpi-4.1.6/bin
prepend-path    LD_LIBRARY_PATH /opt/openmpi-4.1.6/lib
prepend-path    MANPATH /opt/openmpi-4.1.6/share/man
prepend-path    CPATH /opt/openmpi-4.1.6/include
setenv          UCX_TLS ud_verbs
setenv          UCX_NET_DEVICES ibp3s0:1
-------------------------------------------------------------------
```

1. Please use the following commands to test different data sizes for latency and bandwidth, to verify your ideas:
```bash
module load openmpi/ucx-pp
mpiucx -n 2 $HOME/UCX-lsalab/test/mpi/osu/pt2pt/osu_latency
mpiucx -n 2 $HOME/UCX-lsalab/test/mpi/osu/pt2pt/osu_bw
```
2. Please create a chart to illustrate the impact of different parameter options on various data sizes and the effects of different testsuite.
3. Based on the chart, explain the impact of different TLS implementations and hypothesize the possible reasons (references required).

```cpp=
mpiucx -n 2 -x UCX_TLS=all $HOME/UCX-lsalab/test/mpi/osu/pt2pt/ocu_latency
mpicux -n 2 -x UCX_TLS=all $HOME?UCX_lsalab/test/mpi/osu/pt2pt/osu_bw
```

<image src="https://hackmd.io/_uploads/Byay-4trkl.png" width=45%>
<image src="https://hackmd.io/_uploads/H1Bx-4FSJx.png" width=45%>

原本配置`UCX_TLS=ud_verbs`表示使用ud_verbs作為傳輸層協議。

`UCX_TLS=all`可以啟用UCX支援的所有傳輸層協議，包括`sm`(shared memory)、`tcp`(TCP/IP)、`ud`(Unreliable Datagram)、`rc`(Reliable Connection)等。它會嘗試初始化並使用系統中可用的所有傳輸協議，根據環境和硬體支持的情況來選擇最合適的協議。從結果也可以看到在相同資料量的情況下，延遲降低但頻寬提高了。
    
reference：
- https://github-wiki-see.page/m/openucx/ucx/wiki/UCX-environment-parameters?utm_source=chatgpt.com
- https://openucx.readthedocs.io/en/master/faq.html?utm_source=chatgpt.com

### Advanced Challenge: Multi-Node Testing

This challenge involves testing the performance across multiple nodes. You can accomplish this by utilizing the sbatch script provided below. The task includes creating tables and providing explanations based on your findings. Notably, Writing a comprehensive report on this exercise can earn you up to 5 additional points.

- For information on sbatch, refer to the documentation at [Slurm's sbatch page](https://slurm.schedmd.com/sbatch.html).
- To conduct multi-node testing, use the following command:
```
cd ~/UCX-lsalab/test/
sbatch run.batch
```


