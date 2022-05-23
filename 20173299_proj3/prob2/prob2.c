#include <omp.h>
#include <stdio.h>

enum {
  STATIC = 1,
  DYNAMIC,
  GUIDED,
};

static long num_steps = 10000000;
double step;

void calculate_static(int chunk_size) {
  long i;
  double x, pi, sum = 0.0;
  double start_time, end_time;
  step = 1.0 / (double)num_steps;

  start_time = omp_get_wtime();

#pragma omp parallel for schedule(static, chunk_size) private(x) reduction(+ : sum)
  for (i = 0; i < num_steps; i++) {
    x = (i + 0.5) * step;
    sum += +4.0 / (1.0 + x * x);
  }

  pi = step * sum;
  end_time = omp_get_wtime();
  double timeDiff = end_time - start_time;
  printf("Execution Time : %lfms\n", timeDiff);

  printf("pi=%.24lf\n", pi);
}

void calculate_dynamic(int chunk_size) {
  long i;
  double x, pi, sum = 0.0;
  double start_time, end_time;
  step = 1.0 / (double)num_steps;

  start_time = omp_get_wtime();

#pragma omp parallel for schedule(dynamic, chunk_size) private(x) reduction(+ : sum)
  for (i = 0; i < num_steps; i++) {
    x = (i + 0.5) * step;
    sum += +4.0 / (1.0 + x * x);
  }

  pi = step * sum;
  end_time = omp_get_wtime();
  double timeDiff = end_time - start_time;
  printf("Execution Time : %lfms\n", timeDiff);

  printf("pi=%.24lf\n", pi);
}

void calculate_guided(int chunk_size) {
  long i;
  double x, pi, sum = 0.0;
  double start_time, end_time;
  step = 1.0 / (double)num_steps;

  start_time = omp_get_wtime();

#pragma omp parallel for schedule(guided, chunk_size) private(x) reduction(+ : sum)
  for (i = 0; i < num_steps; i++) {
    x = (i + 0.5) * step;
    sum += +4.0 / (1.0 + x * x);
  }

  pi = step * sum;
  end_time = omp_get_wtime();
  double timeDiff = end_time - start_time;
  printf("Execution Time : %lfms\n", timeDiff);

  printf("pi=%.24lf\n", pi);
}

int main(int argc, char *argv[]) {
  // return error if argc is not 4
  if (argc != 4) {
    printf("Usage: ./prob2 [1|2|3] <chunk_size> <number of threads>\n");
    return -1;
  }

  int type = atoi(argv[1]);
  int chunk_size = atoi(argv[2]);
  int num_of_threads = atoi(argv[3]);

  // return error if type is not in the range
  if (type < STATIC || type > GUIDED) {
    printf("Invalid schedule type\n");
    return -1;
  }

  // return error if chunk_size is not 1, 5, 10, 100
  if (chunk_size != 1 && chunk_size != 5 && chunk_size != 10 &&
      chunk_size != 100) {
    printf("Error: chunk_size should be 1,5,10,100\n");
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

  switch (type) {
    case STATIC:
      printf("Static schedule, chunk size = %d\n", chunk_size);
      calculate_static(chunk_size);
      break;
    case DYNAMIC:
      printf("Dynamic schedule, chunk size = %d\n", chunk_size);
      calculate_dynamic(chunk_size);
      break;
    case GUIDED:
      printf("Guided schedule, chunk size = %d\n", chunk_size);
      calculate_guided(chunk_size);
      break;
  }
  return 0;
}