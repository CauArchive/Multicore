#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>

#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono;

long num_steps = 1000000000;
// global variables for host and device
__managed__ double step;

struct each_step {
  // function that could be called by host and device
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

  // declare device vector
  thrust::device_vector<double> x_dev(num_steps);
  // assign values from 0 to num_steps - 1
  thrust::sequence(x_dev.begin(), x_dev.end());

  // transform each element of x_dev to each_step and sum them
  sum = thrust::transform_reduce(x_dev.begin(), x_dev.end(), each_step(), 0.0,
                                 thrust::plus<double>());
  // get pi
  pi = step * sum;

  // stop timer
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  cout << "With CUDA: " << (duration.count() / 1000000.0) << " seconds" << endl;

  printf("pi=%.8lf\n", pi);

  return 0;
}