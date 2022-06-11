#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>

#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono;

long num_steps = 1000000000;
__managed__ double step;

struct each_step {
  __host__ __device__ double operator()(double i) {
    double x = (i + 0.5) * step;
    return 4.0 / (1.0 + x * x);
  }
};

int main() {
  double pi, sum = 0.0;
  step = 1.0 / (double)num_steps;

  // start timer
  auto start = high_resolution_clock::now();

  thrust::device_vector<double> x_dev(num_steps);
  thrust::sequence(x_dev.begin(), x_dev.end());

  sum = thrust::transform_reduce(x_dev.begin(), x_dev.end(), each_step(), 0.0,
                                 thrust::plus<double>());
  pi = step * sum;

  // stop timer
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  cout << "With CUDA: " << (duration.count() / 1000000.0) << " seconds" << endl;

  printf("pi=%.8lf\n", pi);

  return 0;
}