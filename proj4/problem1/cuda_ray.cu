#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <chrono>
#include <iostream>

#define SPHERES 100

#define rnd(x) (x * rand() / RAND_MAX)
#define INF 2e10f
#define DIM 2048

using namespace std;
using namespace std::chrono;

struct Sphere {
  float r, b, g;
  float radius;
  float x, y, z;
  __host__ __device__ float hit(float ox, float oy, float* n) {
    float dx = ox - x;
    float dy = oy - y;
    if (dx * dx + dy * dy < radius * radius) {
      float dz = sqrtf(radius * radius - dx * dx - dy * dy);
      *n = dz / sqrtf(radius * radius);
      return dz + z;
    }
    return -INF;
  }
};

__global__ void kernel(Sphere* s, unsigned char* ptr) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int offset = x + y * blockDim.x * gridDim.x;
  float ox = (x - DIM / 2);
  float oy = (y - DIM / 2);

  float r = 0, g = 0, b = 0;
  float maxz = -INF;
  for (int i = 0; i < SPHERES; i++) {
    float n;
    float t = s[i].hit(ox, oy, &n);
    if (t > maxz) {
      float fscale = n;
      r = s[i].r * fscale;
      g = s[i].g * fscale;
      b = s[i].b * fscale;
      maxz = t;
    }
  }

  ptr[offset * 4 + 0] = (int)(r * 255);
  ptr[offset * 4 + 1] = (int)(g * 255);
  ptr[offset * 4 + 2] = (int)(b * 255);
  ptr[offset * 4 + 3] = 255;
}

void ppm_write(unsigned char* bitmap, int xdim, int ydim, FILE* fp) {
  int i, x, y;
  fprintf(fp, "P3\n");
  fprintf(fp, "%d %d\n", xdim, ydim);
  fprintf(fp, "255\n");
  for (y = 0; y < ydim; y++) {
    for (x = 0; x < xdim; x++) {
      i = x + y * xdim;
      fprintf(fp, "%d %d %d ", bitmap[4 * i], bitmap[4 * i + 1],
              bitmap[4 * i + 2]);
    }
    fprintf(fp, "\n");
  }
}

int main(int argc, char* argv[]) {
  unsigned char* bitmap;
  unsigned char* cu_bitmap;
  Sphere* temp_s;
  Sphere* cu_temp_s;

  srand(time(NULL));

  if (argc != 2) {
    printf("> a.out [filename.ppm]\n");
    printf("for example, '> a.out result.ppm' means executing With CUDA\n");
    exit(0);
  }

  FILE* fp = fopen(argv[1], "w");

  int sphere_size = sizeof(Sphere) * SPHERES;
  temp_s = (Sphere*)malloc(sphere_size);
  for (int i = 0; i < SPHERES; i++) {
    temp_s[i].r = rnd(1.0f);
    temp_s[i].g = rnd(1.0f);
    temp_s[i].b = rnd(1.0f);
    temp_s[i].x = rnd(2000.0f) - 1000;
    temp_s[i].y = rnd(2000.0f) - 1000;
    temp_s[i].z = rnd(2000.0f) - 1000;
    temp_s[i].radius = rnd(200.0f) + 40;
  }

  int bitmap_size = sizeof(unsigned char) * DIM * DIM * 4;
  bitmap = (unsigned char*)malloc(bitmap_size);

  dim3 grids(DIM / 32, DIM / 32);
  dim3 threads(32, 32);

  // start measurement
  auto start = high_resolution_clock::now();

  cudaMalloc((void**)&cu_temp_s, sphere_size);
  cudaMemcpy(cu_temp_s, temp_s, sphere_size, cudaMemcpyHostToDevice);
  cudaMalloc((void**)&cu_bitmap, bitmap_size);
  // ray tracing kernel function
  kernel<<<grids, threads>>>(cu_temp_s, cu_bitmap);

  // copy to main memory
  cudaMemcpy(bitmap, cu_bitmap, bitmap_size, cudaMemcpyDeviceToHost);

  // stop timer
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  cudaFree(cu_bitmap);
  cudaFree(cu_temp_s);

  cout << "With CUDA: " << (duration.count() / 1000000.0) << " seconds" << endl;

  ppm_write(bitmap, DIM, DIM, fp);

  fclose(fp);
  free(bitmap);
  free(temp_s);

  return 0;
}