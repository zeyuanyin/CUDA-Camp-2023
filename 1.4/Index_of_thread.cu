#include <stdio.h>

__global__ void hello_from_gpu()
{
    int bid = blockIdx.x;
    int gid = threadIdx.x;
    printf("Hello World from the GPU! blockIdx is %d, threadIdx is %d!\n", bid, gid);
}

int main(void)
{
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}