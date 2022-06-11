#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  printf("max_thread : %d\n", omp_get_max_threads());
  printf("num_thread : %d\n", omp_get_num_threads());
#pragma omp parallel
  {
    printf("thread_num : %d\n", omp_get_thread_num());
    printf("num_thread : %d\n", omp_get_num_threads());
  }

  return 0;
}