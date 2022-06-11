/**
* Multicore Programming Project03 - Problem 1
Author: Kim, Dong-Wook
Student ID: 20173299
**/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define NUM_STEPS 200000

enum {
  STATIC_DEFAULT = 1,
  DYNAMIC_DEFAULT,
  STATIC_CHUNKED_10,
  DYNAMIC_CHUNKED_10,
};

int prime_check(int n) {
  int i;
  if (n <= 1) return 0;
  for (i = 2; i <= n / 2; i++) {
    if (n % i == 0) return 0;
  }
  return 1;
}

void calculate_static() {
  int counter = 0;

  double start_time, end_time;
  start_time = omp_get_wtime();

#pragma omp parallel for schedule(static)
  {
    for (int i = 1; i <= NUM_STEPS; i++) {
      if (prime_check(i)) {
#pragma omp atomic
        counter++;
      }
    }
  }
  end_time = omp_get_wtime();

  double timeDiff = end_time - start_time;
  // printf("Execution Time : %lfms\n", timeDiff);
  printf("%lf,\n", timeDiff);

  // printf("%d\n", counter);
}

void calculate_static_chunk_10() {
  int counter = 0;

  double start_time, end_time;
  start_time = omp_get_wtime();

#pragma omp parallel for schedule(static, 10)
  {
    for (int i = 1; i <= NUM_STEPS; i++) {
      if (prime_check(i)) {
#pragma omp atomic
        counter++;
      }
    }
  }
  end_time = omp_get_wtime();

  double timeDiff = end_time - start_time;
  // printf("Execution Time : %lfms\n", timeDiff);
  printf("%lf,\n", timeDiff);

  // printf("%d\n", counter);
}

void calculate_dynamic() {
  int counter = 0;

  double start_time, end_time;
  start_time = omp_get_wtime();

#pragma omp parallel for schedule(dynamic)
  {
    for (int i = 1; i <= NUM_STEPS; i++) {
      if (prime_check(i)) {
#pragma omp atomic
        counter++;
      }
    }
  }
  end_time = omp_get_wtime();

  double timeDiff = end_time - start_time;
  // printf("Execution Time : %lfms\n", timeDiff);
  printf("%lf,\n", timeDiff);

  // printf("%d\n", counter);
}

void calculate_dynamic_chunk_10() {
  int counter = 0;

  double start_time, end_time;
  start_time = omp_get_wtime();

#pragma omp parallel for schedule(dynamic, 10)
  {
    for (int i = 1; i <= NUM_STEPS; i++) {
      if (prime_check(i)) {
#pragma omp atomic
        counter++;
      }
    }
  }
  end_time = omp_get_wtime();

  double timeDiff = end_time - start_time;
  // printf("Execution Time : %lfms\n", timeDiff);
  printf("%lf,\n", timeDiff);

  // printf("%d\n", counter);
}

int main(int argc, char *argv[]) {
  // return error if argc is not 3
  if (argc != 3) {
    printf("Usage: ./prob1 [1|2|3|4] <number of threads>\n");
    return -1;
  }

  // parse arguments
  int sch_type = atoi(argv[1]);
  int num_of_threads = atoi(argv[2]);

  // return error if sch_type is not in the range
  if (sch_type < STATIC_DEFAULT || sch_type > DYNAMIC_CHUNKED_10) {
    printf("Error: sch_type is not in the range\n");
    return -1;
  }

  // return error if num_of_threads is not 1,2,4,6,8,10,12,14,16
  if (num_of_threads != 1 && num_of_threads != 2 && num_of_threads != 4 &&
      num_of_threads != 6 && num_of_threads != 8 && num_of_threads != 10 &&
      num_of_threads != 12 && num_of_threads != 14 && num_of_threads != 16) {
    printf("Error: num_of_threads should be 1,2,4,6,8,10,12,14,16\n");
    return -1;
  }

  // set number of threads
  omp_set_num_threads(num_of_threads);

  switch (sch_type) {
    case STATIC_DEFAULT:
      // printf("Static schedule, default chunk size\n");
      calculate_static();
      break;
    case DYNAMIC_DEFAULT:
      // printf("Dynamic schedule, default chunk size\n");
      calculate_dynamic();
      break;
    case STATIC_CHUNKED_10:
      // printf("Static schedule, chunk size = 10\n");
      calculate_static_chunk_10();
      break;
    case DYNAMIC_CHUNKED_10:
      // printf("Dynamic schedule, chunk size = 10\n");
      calculate_dynamic_chunk_10();
      break;
  }
  return 0;
}