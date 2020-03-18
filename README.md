# Game of Life on GPU using CUDA

A EECS 468 project by Yifeng Chen and Junlin Liu

## Compile

```
mkdir build
cd build
cmake ..
make
```

## Run

```
./bin/game_of_life
```

## Method

### Serial CPU Implementation

For the serial CPU implementation, we simply go through every cell in the world, visit their neighbors to count the number of alive cells around it and determine the status of the current cell. The performance of this method is very poor because the cpu need to go over the entire word cell by cell serially to compute the result.

### Naïve GPU Implementation (Byte per cell)

The naïve GPU implementation is very similar to the CPU implementation. Instead of doing a for loop in CPU, we can use multiple threads to compute the state of each cell. Ideally every single thread computes the state of an individual cell. Hence the performance is increased by parallel computing.

In the naïve version of kernel, first we find the position of the cell’s neighbor.

```c++
uint x = cellId % worldWidth;
uint yAbs = cellId - x;
uint xLeft = (x + worldWidth - 1) % worldWidth;
uint xRight = (x + 1) % worldWidth;
uint yAbsUp = (yAbs + worldSize - worldWidth) % worldSize;
uint yAbsDown = (yAbs + worldWidth) % worldSize;
```

In order to minimize the warp divergence, we evaluate each cell by adding the value of its neighbors and compare the result to the rule.\

```c++
uint aliveCells = 
lifeData[xLeft + yAbsUp] + lifeData[x + yAbsUp] + lifeData[xRight + yAbsUp]
 + lifeData[xLeft + yAbs] + lifeData[xRight + yAbs]
 + lifeData[xLeft + yAbsDown] + lifeData[x + yAbsDown] + lifeData[xRight + yAbsDown];

resultLifeData[x + yAbs] = 
aliveCells == 3 || (aliveCells == 2 && lifeData[x + yAbs]) ? 1 : 0;
```

This method significantly improves the performance because every cell can be computed in parallel. However, this method is not ideal because there are a lot of unnecessary read and misaligned read to the global memory.

### Bit per Cell Implementation

We can easily notice that there are only two state for an individual cell: live or dead. So, we do not need to store a cell in a byte, instead we can use the bits in a byte to represent a cell. This approach reduces the size of the data by 8 times because now we can obtain 8 cells with the read of a single byte.

But there is still extra memory read during the evaluation. To evaluate a cell, we need the information of all its neighbors. To evaluate a byte of cells (8 cells), we need to read all it’s surrounding bytes, which means the evaluation a byte of data needs 9 bytes read in total. 

The overhead can be reduced by letting each thread evaluate more than 1 byte of data. Because the cost of evaluate consecutive byte is constant no matter how many consecutive bytes the thread is evaluating. 

But exactly how many bytes should we let a thread to process? If we let a thread to process more bytes, the memory overhead can be reduced, but it somehow reduces the performance because a thread needs more time to process multiple bytes. We will discuss this is the result.

Then we will discuss the detailed implementation. The kernel is still evaluating the state of the cell according to the rule. But now the basic unit of a read is a bit, so we use a for loop to iterate through each bit a evaluate them. In order to test the performance impact of the number of bytes processed by a thread, we added it as the parameter of the kernel.

### Bit per Cell encoding and decoding

To evaluate the data in bits, we need first encode the data from byte representation to bit representation and then decode it. Because there is no data dependency between cells, it’s perfect for parallel in GPU. 

The kernel is very simple and straight forward. Read in the data in a striding pattern, encode 8 byte data into 1 byte and then write it back to the global memory. The decoding process is pretty much the same.

### Bit per Cell with 32 bits read

The bit per cell implementations solves a lot of problem in the naïve implementation and improves the performance. But the are still a significant problem in that approach. The access to global memory is only capable of reading 32/64/128 bits each time. But in bit per cell approach we only read 8 bits a time. In order to read 8 bits, the kernel need to retrieve 32 bits of data, which means we are wasting 3/4 of the memory bandwidth. To tackle this problem, instead of reading a byte each time, we read in a 32-bit uint in each time.

But this introduces extra problem, as showed in Figure 5, the CPU in lab machines is using little endian to store data that is longer than 1 byte. That means if we store the data in a 32-bit int, they will be stored in a reversed order, hence we need to convert the data to big endian when we want to process them. 

After that the process is the same as the bit per cell implementation.

## Result

| World Size | CPU      | Basic GPU  | 1 Byte     | 2 Bytes    | 8 Bytes    | 32 Bytes   | 32 Bits Read |
| ---------- | -------- | ---------- | ---------- | ---------- | ---------- | ---------- | ------------ |
| 65536      | 24453.73 | 182044.44  | 211406.45  | 198593.94  | 177124.32  | 109226.67  | 197609.85    |
| 262144     | 28680.96 | 689852.63  | 728177.78  | 771011.76  | 748982.86  | 416101.59  | 770193.64    |
| 589824     | 28856.36 | 1254944.68 | 1638400.00 | 1594118.92 | 1685211.43 | 867388.24  | 1593863.66   |
| 1048576    | 27865.43 | 1691251.61 | 2759410.53 | 2688656.41 | 2912711.11 | 1519675.36 | 2683070.99   |
| 1638400    | 28875.57 | 2214054.05 | 3485957.45 | 3640888.89 | 3900952.38 | 2374492.75 | 3564658.65   |
| 2359296    | 29271.66 | 2711834.48 | 4369066.67 | 5128904.35 | 5362036.36 | 3370422.86 | 5086865.58   |
| 3211264    | 27559.77 | 2973392.59 | 5734400.00 | 6058988.68 | 6175507.69 | 4587520.00 | 6015925.77   |
| 4194304    | 28284.47 | 3084047.06 | 6452775.38 | 6990506.67 | 7108989.83 | 5907470.42 | 6927162.49   |

