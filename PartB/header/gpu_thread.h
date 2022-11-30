#define tile (32)

// Create other necessary functions here
__global__ void matrix_tile_mul(int *a,int *b,int *c,int N) {
   //Allocate shared memory
    __shared__ int A[tile*tile];
    __shared__ int B[tile*tile];
    
   //calculate Rows and columns
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < N/2 && col < N/2) {
    int threadX = threadIdx.x;
    int threadY = threadIdx.y;
    int dim = blockDim.x;

    //move tile across the length of the grid
    int temp = 0;
    for(int i=0;i<((N+dim-1)/dim);i++) {
      A[threadY*dim + threadX] = a[(2*row*N)+(i*dim)+threadX] + a[((2*row+1)*N)+(i*dim)+threadX];
      B[threadY*dim + threadX] = b[(i*dim*N)+(threadY*N)+2*col] + b[(i*dim*N)+(threadY*N)+2*col+1];
      __syncthreads();

      for(int j=0;j<dim;j++) {
          temp+=(A[threadY*dim + j]*B[j*dim + threadX]);
        __syncthreads();
      }
    }
    //write back result in array
    c[(row*N/2 + col)]=temp;
  }
}

__global__ void matrix_mul(int *a,int *b,int *c,int N) {
  //make thread for one row and col of output matrix
  //retrieve row and col
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int sum = 0;
  for(int i=0;i<N;i+=1) {
    
    sum+= (a[2*row*N + i] + a[(2*row+1)*N + i])*( b[i*N + 2*col] +  b[i*N + (2*col+1)]);
  }
  //write back a result
  c[row*(N/2) + (col)]= sum;
}
// Fill in this function
void gpuThread(int N, int *matA, int *matB, int *output)
{
  int kernel = 0;
  int size = N*N*sizeof(int);
  
  // cudaMallocHost((void **)&h_d,size/4);
  int *d_a,*d_b,*d_c,*d_d;
  cudaMalloc((void **)&d_a,size);
  cudaMalloc((void **)&d_b,size);
  cudaMalloc((void **)&d_c,size/4);
  // cudaMalloc((void **)&d_d,size/4);
  // cudaMalloc((void **)&d_matrix_mul_array,size/4);

  // //transfer host to device
  cudaMemcpy(d_a, matA, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, matB, size, cudaMemcpyHostToDevice);

  int threads = tile;
  int NoOfBlocks = (N/2+threads-1) / threads;

  // //setup kernel
  dim3 THREADS(threads,threads);
  dim3 BLOCKS(NoOfBlocks,NoOfBlocks);

  
  // matrix_mul<<<BLOCKS,THREADS>>>(d_a,d_b,d_d,N);
  // cudaDeviceSynchronize();
  matrix_tile_mul<<<BLOCKS,THREADS>>>(d_a,d_b,d_c,N);
  cudaDeviceSynchronize();

  cudaMemcpy(output, d_c, size/4, cudaMemcpyDeviceToHost);
  // cudaMemcpy(h_d, d_d, size/4, cudaMemcpyDeviceToHost);

  
}
