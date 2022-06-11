#include <omp.h>
#include <stdio.h>

long num_steps = 1000000000;
double step;

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <number of threads>\n", argv[0]);
    return 1;
  }

  int num_threads = atoi(argv[1]);
  if (num_threads > 0) {
    omp_set_num_threads(num_threads);
  } else {
    printf("Invalid number of threads\n");
    return 1;
  }

  long i;
  double x, pi, sum = 0.0;
  step = 1.0 / (double)num_steps;

  double start = omp_get_wtime();

#pragma omp parallel for reduction(+ : sum) private(x)
  for (i = 0; i < num_steps; i++) {
    x = (i + 0.5) * step;
    sum = sum + 4.0 / (1.0 + x * x);
  }
  pi = step * sum;

  double end = omp_get_wtime();

  printf("With OpenMP: %f seconds\n", end - start);
  printf("Pi with %d threads: %f\n", num_threads, pi);
  printf("pi=%.8lf\n", pi);

  return 0;
}