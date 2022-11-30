 #include <immintrin.h>

void singleThread(int N, int *matA, int *matB, int *output)
{

  __m256i vRowA = _mm256_setzero_si256();
  __m256i vRowB = _mm256_setzero_si256();
  __m256i vFinal = _mm256_setzero_si256();
  int *normal = new int[N*N];

  int rowA, colB, iter;
  for (rowA = 0; rowA < N; rowA++)
  {
      for (colB = 0; colB < N; colB++)
      {
        vRowA = _mm256_set1_epi32(matA[rowA * N + colB]);//an element of matA is stored across the vector
        for (iter = 0; iter < N; iter += 8)
        {
          vRowB = _mm256_loadu_si256((__m256i*)&matB[colB * N + iter]); //Transfer row of matB
          vFinal = _mm256_loadu_si256((__m256i*)&normal[rowA * N + iter]); //Loads the result matrix row as a vector
          vFinal = _mm256_add_epi32(vFinal ,_mm256_mullo_epi32(vRowA, vRowB));//Multiplies the vectors and adds to the result vector
          _mm256_storeu_si256((__m256i*) &normal[rowA * N + iter], vFinal);
        }
      }
  }
    
  for(int rowC = 0; rowC < N; rowC++)
  {

    for(int colC = 0; colC < N; colC++)
    {
      int indexC = (rowC >> 1) * (N >> 1) + (colC >> 1);
      int indexOriginal = rowC * N + colC;
      output[indexC] += normal[indexOriginal];
    }
  }
}





