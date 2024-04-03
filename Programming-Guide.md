# Programming Guide

[cuda编程初步入门指导](https://docs.nvidia.com/cuda/archive/12.1.1/cuda-c-programming-guide/index.html)

![The GPU Devotes More Transistors to Data Processing](./image/gpu-devotes-more-transistors-to-data-processing.png)

这个图说的是cpu和gpu的架构差异，侧重点，cpu很多面积用在了缓存和控制上，gpu则强调计算能力，大部分面积用在计算核心上。cpu的多层存储架构和逻辑单元保障通用软件的灵活性。

gpu天生的特点(为绘制图形设计)就是大数据，并行计算能力；

![Automatic Scalability](./image/automatic-scalability.png)

cuda是一个**可伸缩**的编程模型，也就是说thread block可以被安排到任一stream multiprocessor去运行，这个任务规划可以放在运行时，根据当前硬件自动匹配。

## Programming Model编程模型

### kernels核函数`__global__`函数

使用cpp定义，但是运行在gpu上的带cuda扩展语法的函数；

```c++
// Kernel 函数定义，__global__是cuda的扩展
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x; //threadIdx是cuda内置关键字，表示当前thread在block中的索引
    C[i] = A[i] + B[i];
}

int main()
{
    ...
    // <<<1, N>>>也是cuda的扩展，里面的数字分别表示grid dim, block dim
    VecAdd<<<1, N>>>(A, B, C);
    ...
}
```

### Thread Hierarchy(层级)

计算当前thread的索引的时候，需要看gridDim, blockIdx， blockDim, threadIdx，

```c++
// Kernel 定义
__global__ void MatAdd(float A[N][N], float B[N][N],
                       float C[N][N])
{
    int i = threadIdx.x; //__globa__ kernel定义的blockDim是二维的
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    ...
    // 1 个block，block内部包含 N * N * 1 个threads
    int numBlocks = 1; //
    dim3 threadsPerBlock(N, N); //blockDim 二维
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```

上面通过你的kernel函数传入的gridDim, blockDim是需要根据任务量进行计算的。

![Grid of Thread Blocks](./image/grid-of-thread-blocks.png)

每个block包含的thread数量是有限制的，比如gtx1650，7个stream multiprocessor，每个sm包含128个cuda core。虽然只有128个core，但是每个block可以最大支持1024个thread。

**thread block只能在一个sm上执行**，不能跨sm；

thread blocks被规划到sm上，可能是串行也可能是并行，所以thread block之间是一个独立的状态，只有在block内部的thread才具备通讯，协作的可能。比如通过shared memeory共享数据，或者设置同步点(调用`__syncthread()`)来协调运行逻辑。

#### Thread Block Clusters

![Grid of Thread Block Clusters](./image/grid-of-clusters.png)

目前最新的rtx 4090 compute capability=8.9，不支持；像gtx 1650使用的turing架构，compute capability=7.5；

[compute capability查询](https://developer.nvidia.com/cuda-gpus)

**Compute Capability 9.0**开始引入了这个概念，cluster 内的thread blocks 会被保证安排到同一个GPU Processing Cluster (GPC)。

cluster的dim也可以是一维，二维或者三维。通常8个thread block是一个移植性较好的值，但是对于一些sm小于8的gpu来说，这个数值应该进一步减小。`cudaOccupancyMaxPotentialClusterSize`去查询这个最大值。

使用这个特性的时候，有如下两种方式：

- kernel函数增加`__cluster_dims__`修饰
- 使用`cudaLaunchKernelEx`调用kernel函数，需要传递一个`cudaLaunchConfig_t`参数。

cluster内部的thread可以访问这个cluster内部的所有block的shared memory：

### Memory Hierarchy（层级）

1. thread -> local memory 

2. thread block -> shared memory， 生命周期等同于block

3. cluster -> multi block shared memory
4. all thread -> global memory

![Memory Hierarchy](./image/memory-hierarchy.png)

### Heterogeneous Programming（异构编程）

异构编程模型，分为host, device部分，他们之间有各自的memory space，通常是不可直接访问对方的memory；cuda的runtime api负责管理global, constant, texture等gpu memory，也负责在host, device之间传送数据，而c++语言里面的api直接管理host也就是cpu的memory。

__Unified Memory__提供managed memory来桥接host和device的memory空间，managed memory可以被所有的cpu, gpu访问，且访问地址是一致的。这个方式扩展了gpu memory，隐藏了显式的在cpu gpu之间同步数据，并且简化了程序移植工作。

![Heterogeneous Programming](./image/heterogeneous-programming.png)

### Asynchronous SIMT Programming Model(异步SIMT编程模型)

从Ampere架构开始，cuda通过异步SIMT编程模型提供加速内存操作；异步SIMT编程模型为对应的cuda thread定义了对应的异步操作；

异步SIMT编程模型为cuda threads定义了异步barrier的具体行为；

cuda异步操作指的是一个cuda thread初始化数据，然后交给令一个cuda thread执行，在一个组织良好的程序中，一个或者更多个thread使用异步操作，然后在某个地方进行(阻塞)同步。

异步thread总是和初始化异步操作的thread关联，异步操作一般使用synchronization object 来同步操作结束。synchronization object 可以显式的由用户管理或者隐式的由library管理。

synchronization object可以是`cuda::barrier` 或者`cuda::pipeline`，synchronization object可以在不同的thread作用域使用，如下是定义：

| thread scope                            | 含义                               |
| --------------------------------------- | ---------------------------------- |
| cuda::thread_scope::thread_scope_thread | 只有初始化异步操作的thread进行同步 |
| cuda::thread_scope::thread_scope_block  | 同一thread block内的thread都同步   |
| cuda::thread_scope::thread_scope_device | 同一gpu上的thread都同步            |
| cuda::thread_scope::thread_scope_system | 同一系统上的gpu thread都同步       |

### Compute Capability

也叫SM version

gtx 1650使用的turing架构，compute capability=7.5；

[compute capability查询](https://developer.nvidia.com/cuda-gpus)

*Hopper GPU* architecture = SM9

Ampere GPU architecture = SM8

*Turing* GPU architecture  = SM7.5

*Volta GPU* architecture = SM7

*Pascal* architecture = SM6

*Maxwell* architecture = SM5

 *Kepler* architecture = SM3

## Programming Interface

cuda c++包含一个规模很小的语法扩展和一个runtime库；任何带有语法扩展的源文件都必须使用nvcc编译；

runtime库包含了：

- 用于打开关闭gpu设备，cpu执行的函数
- 在cpu, gpu之间传输数据的函数
- 管理多gpu的系统函数

runtime库建立在底层c函数和cuda驱动api之上，这些函数也可以被应用程序访问到；

驱动程序应用程序接口通过公开 CUDA context（类似于device的host处理）和 CUDA 模块（类似于device的动态加载库）等低级概念，提供额外的控制级别。

大部分情况下应用都不需要cuda 驱动api，这种情况下写代码更简洁。同时在需要的时候也可以访问使用驱动api。

### nvcc编译

kernel函数可以使用cuda特定架构指令集(PTX)编写，但通常更高层的c++更容易编写。

high-level的c++代码需要使用nvcc编译处理生成binary code。

#### 离线编译

nvcc能够编译host code和device code的混合代码文件，首先分离device code出来然后：

- 把device code编译为汇编PTX代码或者binary文件
- 修改host code，把`<<<...>>>`扩展的kernel函数替换为cuda runtime函数，从PTX或者binary加载调用编译好的kernel函数

修改后的host code将使用另一个编译器编译，或者让nvcc在编译最后阶段调用host编译器去编译；

application然后就可以链接到编译后的host code，生成可执行文件（大部分情况下都是这样）；或者可以使用cuda驱动api执行PTX或者binary文件。

#### JIT编译

前面我们提到过application可以通过cuda 驱动api加载PTX代码，在运行时阶段PTX代码会被进一步编译成binary，这就是just-in-time编译；JIT编译减慢了加载速度，但可以让程序从更新驱动，编译器之后获利；

设想这样的情况，开发编译阶段你的device型号是A，但是你的程序将来要分发到device 型号B的机器上，JIT通常也是唯一的一条路；

device驱动JIT编译PTX代码通常会生成一个cache，用来避免重复编译，当device 驱动更新之后，这个cache也将失效，这样就能通过更新驱动的方式提升性能。

#### 二进制兼容

二进制代码是和架构绑定的，nvcc使用`-code=sm_75`选项指定，sm_75表示为compute capability = 7.5的gpu生成二进制程序，最简单的兼容规则是， `-code=sm_75`生成的二进制程序可以运行在compute capability >= 7.5的gpu上；但反过来不行；

#### PTX兼容

nvcc的`-arch=compute_75`选项用来指定当编译c++代码到PTX代码时的compute capability版本，因为某些PTX指令只有在高版本compute capability才支持，所以如果你的c++代码包含某些高版本的compute capability才支持的指令，就必须显式地指定该选项。

指定-arch的PTX代码通常都可以编译到对应或者更高版本sm的二进制程序；注意，从低版本compute capability编译而来的PTX代码可能没有用到最新的硬件特性，比如从-arch=compute_60生成的PTX代码，然后编译到-code=sm_70的二进制程序，就不会用到Tensor Core的指令；

#### Application兼容

要在一个特定的compute capability的gpu上执行，这个程序必须从二进制代码加载或者JIT编译PTX代码，相应的二进制或者PTX兼容需要满足要求；如果要让这个程序可以在将来更高版本的compute capability gpu上执行，则必须使用程序加载JIT编译PTX的方式进行；

在程序内部嵌入PTX和二进制代码的方式是使用`-gencode`的`arch`和`code`子选项：

```bash
nvcc x.cu
        -gencode arch=compute_50,code=sm_50
        -gencode arch=compute_60,code=sm_60
        -gencode arch=compute_70,code=\"compute_70,sm_70\"
```

这个例子里面前两个-gencode嵌入sm_50, sm_60两种二进制代码，第三个-gencode嵌入sm_70的PTX和二进制代码；

cpu侧在运行时自动根据当前compute capability 版本选择最近的gpu 代码执行；

比如-code=sm_50生成的二进制代码会在compute capability=5.0, 5.2的gpu上执行，-code=sm_60生成的二进制代码会在compute capability=6.0, 6.1的gpu上执行，-code=sm_70生成的二进制代码会在compute capability=7.0, 7.5的gpu上执行，内嵌的PTX代码可以在compute capability=8.0, 8.6的gpu上JIT编译执行；

nvcc通过`-arch`指定`__CUDA_ARCH__`宏，从而可以在device代码中根据arch版本写不同的代码实现；

`arch=compute_80` 指定的 `__CUDA_ARCH__`=`800`；

如果直接使用驱动API，则必须编译到不同的文件，在运行时自行指定启动哪个文件；

Volta架构引入了*Independent Thread Scheduling*来改变gpu的thread映射，如果要强制使用之前版本的SIMT scheduling，则可以使用`-arch=compute_60 -code=sm_70`

`-arch=sm_70` 是 `-arch=compute_70 -code=compute_70,sm_70` 的简写，同时也等价于 `-gencode arch=compute_70,code=\"compute_70,sm_70\"`

#### C++兼容

host代码是完全兼容标准c++的，device代码则使用c++的一个子集；

#### x64兼容

64bit的nvcc在64bit模式下编译device代码，指针size是64 bit，64bit模式下编译的device 代码要求host端也要是64bit模式；

### CUDA runtime

静态库：cudart.a或者cudart.lib，动态库：cudart.so或者cudart.dll, 里面所有的接口都有前缀cuda;

#### 初始化

使用`cudaInitDevice()`和`cudaSetDevice()`初始化runtime和主上下文环境，如果不手动调用，则runtime隐式使用device 0和自动初始化，来处理其他runtime api; **注意**，在12.0之前，`cudaSetDevice()`不会初始化runtime，而应用程序中经常在激活其他api前使用`cudaFree()`来隔离runtime初始化，所以这里对于测量timing和解释第一个错误码的时候有一定的异常风险；总之，手动调用这两个api总不会错的。

cuda runtime为每个gpu device创建一个cuda context，这个context是该device的主context，并在第一个需要激活context的api调用前初始化完成。context在host端的所有threads之间共享，`__device__`的代码编译为PTX后，如果有必要的话，会进行JIT编译，然后在gpu memory中加载，当然在runtime 和驱动api混合使用的时候，驱动api也可以访问到这个context。

当cpu端使用`cudaDeviceReset()`后，会销毁这个主context，在host端下一个runtime api被调用前，一个新的主context会被创建出来；

#### device memory

kernel不负责操作device memory，所以都是在host端进行device memory管理，创建，销毁，复制数据等；

cuda arrays: 不透明的内存分布，为texture访问深度优化；

linear memory：一个统一的地址空间，可以使用指针访问，比如b-tree, 链表等数据节构内部，大量的指针访问；寻址空间最大值，依赖host和compute capability;

|                        | x86_64    | POWER     | ARM64     |
| ---------------------- | --------- | --------- | --------- |
| compute capability 5.3 | 40bit     | 40bit     | 40bit     |
| compute capability 6.0 | 最大47bit | 最大49bit | 最大48bit |

linear memory的典型使用是`cudaMalloc()`和`cudaFree()`，在host和device之间传输数据的`cudaMemcpy()`。令外，linear memory也可以使用`cudaMallocPitch()`和`cudaMalloc3D()`，他们适合用来给2D或者3D数组申请内存，为了提高性能，会在数据行末自动padding做内存对齐，这样在访问row数据或者数据复制(`cudaMemcpy2D()`和`cudaMemcpy3D()`)的时候会得到最大性能，必须使用api返回的pith(stride步长)来访问数据：

```c++
// Host code
int width = 64, height = 64;
float* devPtr;
size_t pitch;
cudaMallocPitch(&devPtr, &pitch, width * sizeof(float), height);
MyKernel<<<100, 512>>>(devPtr, pitch, width, height);

// Device code
__global__ void MyKernel(float* devPtr, size_t pitch, int width, int height)
{
    for (int r = 0; r < height; ++r)
    {
        float* row = (float*)((char*)devPtr + r * pitch);
        for (int c = 0; c < width; ++c)
        {
            float element = row[c];
        }
    }
}
```

如下列举了一些访问device global memory的方法：

```c++ 
__constant__ float constData[256];
float data[256];
cudaMemcpyToSymbol(constData, data, sizeof(data));
cudaMemcpyFromSymbol(data, constData, sizeof(data));

__device__ float devData;
float value = 3.14f;
cudaMemcpyToSymbol(devData, &value, sizeof(float));

__device__ float* devPointer;
float* ptr;
cudaMalloc(&ptr, 256 * sizeof(float));
cudaMemcpyToSymbol(devPointer, &ptr, sizeof(ptr));
```

`cudaGetSymbolAddress()` 用于获取global memory变量的地址，`cudaGetSymbolSize()`获取其大小。

#### Device Memory L2 访问管理

定义重复访问global memory的某一个数据为persisting， 只访问一次的叫streaming；

cuda 11.0, compute capability 8.0开始支持把persisting数据放置到L2缓存中，提高带宽和减小时延；

可以设置留存一部分L2缓存给presisting数据访问，presisting访问优先使用该区域，只有当presisting未使用时，streaming或者global访问才会使用该区域；

```c++
cudaGetDeviceProperties(&prop, device_id);
size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); /* 留存 3/4 of L2 缓存给 persisting 访问*/
```

**注意**，当GPU 被配置未Multi-Instance GPU (MIG) 模式时, 这个功能不可用；如果使用Multi-Process Service (MPS)，`cudaDeviceSetLimit`失效，设置PersistingL2CacheSize的容量时环境变量`CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT`决定的。

##### persisting访问政策管理

简单来说就是指定global memory中的一块连续内存地址，再配置一个persistence property：

cuda stream 示例：

```c++
cudaStreamAttrValue stream_attribute;// Stream level attributes data structure
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr); // Global Memory data pointer
stream_attribute.accessPolicyWindow.num_bytes = num_bytes;                    // Number of bytes for persistence access.
                                                                              // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
stream_attribute.accessPolicyWindow.hitRatio  = 0.6;                          // Hint for cache hit ratio
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

//Set the attributes to a CUDA stream of type cudaStream_t
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
```

ptr到ptr+num_bytes之间的数据访问就会有大概率放置到L2访问，而不是访问global memory；

cuda graph kernel node 示例：

```c++
cudaKernelNodeAttrValue node_attribute;                                     // Kernel level attributes data structure
node_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr); // Global Memory data pointer
node_attribute.accessPolicyWindow.num_bytes = num_bytes;                    // Number of bytes for persistence access.
                                                                            // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
node_attribute.accessPolicyWindow.hitRatio  = 0.6;                          // Hint for cache hit ratio
node_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
node_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

//Set the attributes to a CUDA Graph Kernel node of type cudaGraphNode_t
cudaGraphKernelNodeSetAttribute(node, cudaKernelNodeAttributeAccessPolicyWindow, &node_attribute);
```

`hitRatio`参数指定了命中概率，上面指定的0.6表示在ptr到ptr+num_bytes之间的数据有60%可能性被划分为persisting属性，而剩下的划分为streaming属性；memory访问使用随机的方式逼近hitRatio这个概率，这个可能性依赖硬件架构和memory extent;

举例来说，加入你设置的L2留存空间时16KB，但是你的num_bytes是32KB，如果`hitRatio=0.5`则会随机选择其中一段作为persisting数据并缓存到L2留存区域；但是，如果`hitRatio=1.0`的话，硬件尝试缓存32KB数据到16KB看留存空间，这样的话会导致cache line抖动，只保持最近使用的16KB数据缓存到留存区，同时增大了L2缓存数据输入和输出；

同时还需要考虑到L2缓存是core共享的，如果设置16KB留存，然后num_bytes也是16KB，设置hitRatio=1.0也会导致问题，因为core并发kernel，如果多个core执行的话，也会导致挤压对方的persisting cache line;

##### L2访问properties

定义了3种property:

cudaAccessPropertyStreaming：使用 streaming 属性发生的内存访问不太可能保留在 L2 缓存中，因为这些访问会优先逐出。

cudaAccessPropertyPersisting：使用 persisting 属性发生的内存访问更有可能保留在 L2 缓存中，因为这些访问优先保留在 L2 缓存的预留部分。

cudaAccessPropertyNormal：此访问属性将以前应用的持久访问属性强制重置为正常状态。具有先前 CUDA 内核的持久化属性的内存访问可能会在其预期用途后很长一段时间内保留在 L2 缓存中。这种使用后持久化减少了不使用 persisting 属性的后续内核可用的 L2 缓存量。使用 cudaAccessPropertyNormal 属性重置访问属性窗口将删除先前访问的持久性（优先保留）状态，就好像先前访问没有访问属性一样。

##### L2持久访问示例

```c++
cudaStream_t stream;
cudaStreamCreate(&stream);                                                                  // Create CUDA stream

cudaDeviceProp prop;                                                                        // CUDA device properties variable
cudaGetDeviceProperties( &prop, device_id);                                                 // Query GPU properties
size_t size = min( int(prop.l2CacheSize * 0.75) , prop.persistingL2CacheMaxSize );
cudaDeviceSetLimit( cudaLimitPersistingL2CacheSize, size);                                  // set-aside 3/4 of L2 cache for persisting accesses or the max allowed

size_t window_size = min(prop.accessPolicyMaxWindowSize, num_bytes);                        // Select minimum of user defined num_bytes and max window size.

cudaStreamAttrValue stream_attribute;                                                       // Stream level attributes data structure
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(data1);               // Global Memory data pointer
stream_attribute.accessPolicyWindow.num_bytes = window_size;                                // Number of bytes for persistence access
stream_attribute.accessPolicyWindow.hitRatio  = 0.6;                                        // Hint for cache hit ratio
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;               // Persistence Property
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;                // Type of access property on cache miss

cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   // Set the attributes to a CUDA Stream

for(int i = 0; i < 10; i++) {
    cuda_kernelA<<<grid_size,block_size,0,stream>>>(data1);                                 // This data1 is used by a kernel multiple times
}                                                                                           // [data1 + num_bytes) benefits from L2 persistence
cuda_kernelB<<<grid_size,block_size,0,stream>>>(data1);                                     // A different kernel in the same stream can also benefit
                                                                                            // from the persistence of data1

stream_attribute.accessPolicyWindow.num_bytes = 0;                                          // Setting the window size to 0 disable it
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   // Overwrite the access policy attribute to a CUDA Stream
cudaCtxResetPersistingL2Cache();                                                            // Remove any persistent lines in L2

cuda_kernelC<<<grid_size,block_size,0,stream>>>(data2);                                     // data2 can now benefit from full L2 in normal mode
```

##### 复位L2访问

更早之前的cuda kernel产生的persisting L2缓存可能持续存在于L2中，因此，必须重置L2缓存到streaming或者normal 内存访问状态，有3中办法：

1. Reset a previous persisting memory region with the access property, `cudaAccessPropertyNormal`.
2. Reset all persisting L2 cache lines to normal by calling `cudaCtxResetPersistingL2Cache()`.
3. **Eventually** untouched lines are automatically reset to normal. Reliance on automatic reset is strongly discouraged because of the undetermined length of time required for automatic reset to occur.

##### 管理L2留存区使用情况

多个kernels在多个stream中并发，他们可能有各自的访问规则配置，但是，L2留存缓存选项是所有kernels共享的，所以，缓存使用情况是所有的kernel使用情况之和；将内存访问指定为持久性访问的好处会随着持久性访问量超过预留的 L2 缓存容量而减少。

工程师需要考虑如下：

- Size of L2 set-aside cache.
- CUDA kernels that may concurrently execute.
- The access policy window for all the CUDA kernels that may concurrently execute.
- When and how L2 reset is required to allow normal or streaming accesses to utilize the previously set-aside L2 cache with equal priority.

##### 查询L2缓存properties

cudaGetDeviceProperties --> cudaDeviceProp，这个属性包含了L2缓存信息：

- `l2CacheSize`: The amount of available L2 cache on the GPU.
- `persistingL2CacheMaxSize`: The maximum amount of L2 cache that can be set-aside for persisting memory accesses.
- `accessPolicyMaxWindowSize`: The maximum size of the access policy window.

##### 控制L2留存缓存

查询`cudaDeviceGetLimit`，设置`cudaDeviceSetLimit`和`cudaLimit`，可设置的最大限制是`cudaDeviceProp::persistingL2CacheMaxSize`

#### Shared Memory

`__shared__`前缀标记的变量表示存储在shared memory的变量，shared memory的访问速度远超global memory，它可以用作暂存器内存（或软件管理的缓存），以最大程度地减少来自 CUDA 块的全局内存访问，如以下矩阵乘法示例所示。

看一个矩阵乘法的示例：

![matrix-multiplication-without-shared-memory.png](./image/matrix-multiplication-without-shared-memory.png)

```c++
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; 
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width; 
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width;
    d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e]
                * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}
```

这个例子很简单，每个thread只负责计算结果矩阵中的一个分量：
$$
C = A * B
$$
A的维度是m行n列， B的维度是n行k列，C的维度是m行k列
$$
C_{i,j} = \sum^{x<n}_{x=0}{A_{i, x}}*{B_{x, j}}
$$
每一个thread都需要从global memory中访问A, B矩阵的数据n次；使用shared memory的方法如下：

![matrix-multiplication-with-shared-memory.png](./image/matrix-multiplication-with-shared-memory.png)

每个block计算一个子区域的结果，并且把对应的A, B子数据先搬到block的shared memory，然后计算过程就不再需要访问global memory；注意线程同步；

```c++
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;
// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}
// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           float value)
{
    A.elements[row * A.stride + col] = value;
}
// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}
// Thread block size
#define BLOCK_SIZE 16
// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);
// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width;
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = B.width;
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width; 
    d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}
// Matrix multiplication kernel called by MatMul()
 __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}
```

#### 分布式shared memory

这个特性需要compute capability 9.0，暂不关注

#### Page-Locked Host Memory

host端的page-locked memory也叫pinned memory，和使用malloc()分配的pageable memory相反;

`cudaHostAlloc()`和`cudaHostFree()`负责分配和释放page-locked memory；

`cudaHostRegister()`可以把`malloc()`分配的内存做page-locks；

使用page-locked有以下好处：

- 在host page-locked memory和device memory之间传输数据可以并发进行；
- 在某些设备上，page-locked memory的地址可以映射到gpu memory地址，这样就可以不需要host-device之间传输数据；
- 在某些具备front-side bus总线的系统上，page-locked memory和device memory之间的带宽会边高，另外如果分配memory有write-combining，会更高；

**Portable Memory:**

cpu控制的page-locked memory可以和系统上任何device协作，但是默认情况下，仅在cpu分配这个memory时的device可以利用page-locked优势，因此，如果想让page-locked memory被所有device所利用，需要给`cudaHostAlloc()`传递`cudaHostAllocPortable`，或者给`cudaHostRegister()`传递`cudaHostRegisterPortable`

**Write-Combining Memory**

默认情况下，page-locked host memory被分配为可缓存。可以选择将其分配为Write-Combining，通过将标志 `cudaHostAllocWriteCombined` 传递给 `cudaHostAlloc()`。Write-Combining内存释放了主机的 L1 和 L2 缓存资源，从而为应用程序的其余部分提供了更多缓存。此外，在通过 PCI Express 总线传输时，写入组合内存不会被窥探，这可以将传输性能提高多达 40%。

从host的Write-Combining内存中读取速度非常慢，因此Write-Combining内存通常应用于host仅写入的内存。

使用CPU atomic 指令on Write-Combining memory 应该避免，因为不是所有的CPU都有实现保证。

**Mapped Memory**

page-locked块内存地址可以被映射到device 内存地址，通过传递`cudaHostAllocMapped`到`cudaHostAlloc()`，或者传递`cudaHostRegisterMapped`到`cudaHostRegister()`。这样的host内存块有两个地址：一个是`cudaHostAlloc()`或者`malloc()`返回的地址，这个是给host端直接使用的；另一个是device内存地址，可以通过`cudaHostGetDevicePointer()`来获取，这个地址可以给kernel使用，从而实现kernel函数访问host 内存块；唯一的一个例外是：当host和device使用了Unifiled Virtual Address Space，这个时候`cudaHostAlloc()`分配的page-locked内存就不能映射到device地址；

从kernel直接访问host内存，它的带宽远远不如device memory带宽大，但这样使用还是有以下好处：

1. 不再需要在device端分配一个memory block并且显式的在host device之间传递数据，这个过程由kernel自己完成
2. 不需要使用stream来重叠优化数据传输和kernel执行过程，kernel自动完成这个重叠；

但是，由于映射的page-locked内存在主机和设备之间共享，因此应用程序必须使用stream或事件同步内存访问，以避免任何潜在的先写后写、先写后写或先写后写危险。

为了能够获取指向任何映射的page-locked内存的设备指针，在执行任何其他 CUDA 调用之前，必须通过使用 `cudaDeviceMapHost` 标志调用 `cudaSetDeviceFlags()` 来启用页面锁定内存映射。否则，`cudaHostGetDevicePointer()` 将返回错误。

`cudaHostGetDevicePointer()` 返回一个错误，如果device不支持映射 page-locked host 内存。 通过查询`canMapHostMemory` 属性可以知道是否支持，1 表示支持映射page-locked host memory；

请注意，从主机或其他设备的角度来看，在映射的页锁定内存上操作的原子函数不是原子操作。

CUDA runtime要求对host内存具备1，2， 4， 8byte自然对齐的加载写入数据保留为单次访问；在某些平台上，硬件可能会将原子分解为单独的加载和存储操作。这些组件加载和存储操作对保留自然对齐的访问具有相同的要求。例如，CUDA 运行时不支持 PCI Express 总线拓扑，其中 PCI Express 网桥将设备和主机之间的 8 字节自然对齐写入拆分为两个 4 字节写入。

#### Memory同步域

某些 CUDA 应用程序可能会看到性能下降，因为内存围栏/刷新操作等待的事务多于 CUDA 内存一致性模型所需的事务。

如下例子：

```c++
__managed__ int x = 0;
__device__  cuda::atomic<int, cuda::thread_scope_device> a(0);
__managed__ cuda::atomic<int, cuda::thread_scope_system> b(0);
// Thread1 (SM)
x = 1;
a = 1;
// Thread2 (SM)
while (a != 1) ;
assert(x == 1);
b = 1;
// Thread3 (CPU) 这里不能通过查询a的状态来决定x==1
while (b != 1) ;
assert(x == 1);
```

上面的例子中`assert`语句都会成立，`x = 1`这个赋值语句先于`b = 1`生效；

因为a是device scope的，所以查询a的值仅仅只能在Thread2中保证x值正确可见，而不能保证在cpu thread3中正确可见；b是system scope的，所以需要保证不仅仅thread2的写入对thread3可见，同时需要保证其他thread的写入对thread2可见，这就是所知的内存同步的累加性，因为gpu在运行时无法知道哪些写入在源头被保证可见，哪些写入是通过timing切换可见，所以gpu会采用保守的大网络来应对无法准确感知的内存操作；

在一些情况下，就导致了干扰：因为GPU不是被要求在源头层面上在等待memory操作结束，所以内存围栏，刷新就会产生额外的耗时。

注意内存围栏可能发生在代码的内联或者原子部分，像上面的例子；或者发生在任务边缘隐藏的完成关系同步中；

一个典型的例子就是，当一个kernel正在本地GPU memory运行的时候，一个并发的kernel（比如从NCCL过来的）执行对等通信。本地的kernel完成运行后，将会隐式的刷新输出来满足与下游工作的关系同步。等待慢速的通讯kernel通过nvlink或者PCIE写入结果，这个过程是完全或部分的非必要的。

##### Isolating Traffic with Domains

需要Hopper架构，暂无法测试；

##### Using Domains in CUDA

需要Hopper架构，暂无法测试；

#### 异步并发执行

CUDA把如下操作视为可以和其他操作并发运行的独立任务：

- host计算
- device计算
- host传输数据到device
- device传输输入到host
- 一个device memory内部数据传输
- device之间memory数据传输

并发的层次依赖到具体的device特性和compute capability。

##### host/device之间的并发执行

host的并发执行是通过异步函数库实现的，异步函数在device完成下发的任务之前就把控制权返还给host。使用异步调用，多个device操作就可以放在device任务队列中，等待device在资源可用的时候去执行。这样就把host从持续管理device中解脱出来，可以去做其他操作。如下的device操作对于host来说是异步调用：

- kernel launches;
- 单个device内部的数据复制
- host/device之间小于等于64KB的memory block复制
- `Async`前缀的memory复制函数调用
- memory set函数调用

开发者可以设置环境变量`CUDA_LAUNCH_BLOCKING=1`来关闭当前系统上所有的CUDA应用去异步执行kernel launches，通常用于开发环境来debug程序，不要再生产环境下设置这个值；

如果硬件计数是通过调试器收集的，则kernel launches是异步执行，除非打开了并发kernle profiling，它就变成同步；带`Async`的memory复制也是异步的，如果host的内存不是page-locked的话；

##### 并发kernel执行

compute capability 2.x或更高版本的部分device可以并发执行kernel，开发者通过查询`concurrentKernels`属性来查看当前硬件是否支持；

device可以支持的最大kernel并发数，需要查询[这个表格](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-c-programming-guide/index.html#features-and-technical-specifications-technical-specifications-per-compute-capability)；

kernel之间的CUDA上下文需要一致，不可以一个kernel带一个不同的CUDA上下文环境；

使用大量device local memory或者texture的kernel可以和其他kernel组成并发执行的机会会降低；

##### 数据传输和kernel执行交织重叠

部分device是可以在执行kernel的时候异步执行memory复制的，开发者通过查询`asyncEngineCount>0`属性可以获取到是否支持，如果涉及到Host memory，则要求一定是page-locked的；

当然device内部数据复制（或者device吞吐数据）和多个kernel并发，组成异步执行也是可能的，device内的数据复制和初始化，使用标准memory copy函数，src和dst地址都是同一个device的memory地址；

##### 数据传输并发

compute capability 2.x或更高版本的部分device可以并发执行数据吞/吐，通过查询属性`asyncEngineCount=2`获取，如果涉及到Host memory，则要求一定是page-locked的；

##### stream

app通过stream来管理上述的并发操作。一个stream表示一个由若干命令（可能由不同的host thread发起）组成的队列，将来他们排队执行。另一方面，不同的stream可能会相对彼此来说无序地执行它们的命令，也可能同时执行它们的命令。这个特征是无法保证具体的行为的，所以不要依赖其正确性（比如，kernel间的通讯)。一个stream发起的命令将在其依赖都准备完毕后执行，其依赖可能是同一stream中的，也可能是其他stream中的。正确的异步调用会保证所有的command都正确执行完毕。

**创建&销毁stream**

通过创建stream对象，并把其作为stream参数传给kernel和memory复制函数；

```c++
cudaStream_t stream[2];
for (int i = 0; i < 2; ++i)
    cudaStreamCreate(&stream[i]);
float* hostPtr;
cudaMallocHost(&hostPtr, 2 * size); //page-locked
```

然后stream就可以通过如下方式，定义一串memory copy to, kernel launches, memory copy from：

```c++
for (int i = 0; i < 2; ++i) {
    cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size,
                    size, cudaMemcpyHostToDevice, stream[i]);
    MyKernel <<<100, 512, 0, stream[i]>>>
          (outputDevPtr + i * size, inputDevPtr + i * size, size);
    cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size,
                    size, cudaMemcpyDeviceToHost, stream[i]);
}
```

每一个stream只负责它管辖的memory，kernel以及结果返回，这样两个stream之间就组成了交织重叠运行，具体的行为依赖device的compute capability。注意，`hostPtr`必须是page-locked的，对于任何overlap交织运行的情况；

销毁stream：

```c++
for (int i = 0; i < 2; ++i)
    cudaStreamDestroy(stream[i]);
```

为了防止在device工作期间，调用了`cudaStreamDestroy`，这个函数会立即返回控制权，并且在device完成所有的任务之后，与这个stream相关联的资源才会自动释放；

**默认stream**

最早的时候我们启动kernel没有配置任何stream参数，或者说，我们隐式的把其设置为0，即从默认stream发起；因此，默认的执行都是固定顺序的；

对于使用了`--default-stream per-thread`编译flag的代码来说，默认stream是一个常规stream，所有的host thread都持有他们自己的默认stream；

使用`--default-stream legacy`编译，默认stream是一个特殊的stream，我们称之为NULL stream，每一个device都有一个，并且应用到说有的host threads；特殊之处在于，它会隐式的做同步动作；

对于没有使用`--default-stream`的编译情况，`--default-stream legacy`是默认flag

**显式手动同步**

`cudaDeviceSynchronize()`同步等待，直到所有的host thread的所有stream完成执行；

`cudaStreamSynchronize()`同步等待，直到指定的stream完成所有操作；这个函数也可用于host和一个指定的stream之间进行同步，其他的stream继续执行；

`cudaStreamWaitEvent()`指定一个stream和event，直到指定的event完成以后，stream所附加的command才会执行；

`cudaStreamQuery()`给host端查询steam内的指令是否已经完成；

**隐式同步**

不同的stream之间的两个command，如果碰到如下情况，则会发生隐式同步：

- 分配page-locked host内存
- 分配device 内存
- device memory set
- device内的memory复制
- 任何NULL stream的CUDA command
- L1/shared memory配置切换

需要确认依赖的操作包括任何在同一个stream内的kernel launch和`cudaStreamQuery`调用，所以app应当遵循如下guide来优化潜在的并发执行kernel：

- 所有的独立操作应当由依赖他们的操作引发
- 同步操作应当尽可能地晚

**重叠行为**

两个流之间的执行重叠量取决于向每个流发出命令的顺序，以及device是否支持数据传输和kernel执行的重叠；

举例来说，在不支持并发传输数据的device上，这段之前的代码：

```c++
for (int i = 0; i < 2; ++i) {
    cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size,
                    size, cudaMemcpyHostToDevice, stream[i]);
    MyKernel <<<100, 512, 0, stream[i]>>>
          (outputDevPtr + i * size, inputDevPtr + i * size, size);
    cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size,
                    size, cudaMemcpyDeviceToHost, stream[i]);
}
```

是完全没有重叠执行的，因为stream[1]的memory host2device只有的stream[0]的memory device2host之后才会引发，所以只有等到stream[0]完成以后，stream[1]才开始执行；但是如果改写成如下：

```c++
for (int i = 0; i < 2; ++i)
    cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size,
                    size, cudaMemcpyHostToDevice, stream[i]);
for (int i = 0; i < 2; ++i)
    MyKernel<<<100, 512, 0, stream[i]>>>
          (outputDevPtr + i * size, inputDevPtr + i * size, size);
for (int i = 0; i < 2; ++i)
    cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size,
                    size, cudaMemcpyDeviceToHost, stream[i]);
```

假如device支持数据传输与kernel执行并发的话，则stream[0]和stream[1]会产生重叠运行，stream[1]的memory host2device会和stream[0]的kernel运行在同一时间运行；

假如device支持数据传输并发，则前面的写法也会产生重叠，stream[1]的HostToDevice会和stream[0]的DeviceToHost产生交织重叠，甚至，假如支持数据传输与kernel执行并发的话，和stream[0]的kernel运行重叠；

**host回调函数**

