#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void collect_to_bucket(int n, int *bucket, int *key) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    __syncthreads();
    atomicAdd(&bucket[key[i]], 1);
  }
  // bucket[key[i]]++;
}
__global__ void collect_to_key(int range, int *bucket, int *key) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < range) {
    int start;
    if (i == 0) {
      start = 0;
    } else {
      start = bucket[i - 1];
    }
    int end = bucket[i];
    for (int j = start; j < end; j++) {
      key[j] = i;
    }
  }
}

__global__ void prefix_sum(int *a, int *b, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (int j = 1; j < N; j <<= 1) {
    b[i] = a[i];
    __syncthreads();
    if (i >= j)
      a[i] += b[i - j];
    __syncthreads();
  }
}

int main() {
  int n = 50;
  int range = 5;
  const int threads_per_block = 64;
  const int blocks = (n + threads_per_block - 1) / threads_per_block;
  const int blocks_b = (range + threads_per_block - 1) / threads_per_block;
  int *g_bucket;
  int *tmp;
  int *key;
  cudaMallocManaged(&g_bucket, range * sizeof(int));
  cudaMallocManaged(&tmp, range * sizeof(int));
  cudaMallocManaged(&key, n * sizeof(int));
  cudaMemset(g_bucket, 0, range * sizeof(int));
  for (int i = 0; i < n; i++) {
    key[i] = rand() % range;
    printf("%d ", key[i]);
  }
  printf("\n");
  cudaDeviceSynchronize();
  collect_to_bucket<<<blocks, threads_per_block>>>(n, g_bucket, key);
  cudaDeviceSynchronize();
  prefix_sum<<<blocks_b, threads_per_block>>>(g_bucket, tmp, range);
  cudaDeviceSynchronize();
  collect_to_key<<<blocks_b, threads_per_block>>>(range, g_bucket, key);
  cudaDeviceSynchronize();

  for (int i = 0; i < n; i++) {
    printf("%d ", key[i]);
  }
  printf("\n");
}
