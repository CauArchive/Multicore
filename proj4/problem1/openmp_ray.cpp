#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SPHERES 500

#define rnd(x) (x * rand() / RAND_MAX)
#define INF 2e10f
#define DIM 2048

// Sphere data structure
typedef struct Sphere {
  float r, b, g;
  float radius;
  float x, y, z;
  float hit(float ox, float oy, float* n) {
    float dx = ox - x;
    float dy = oy - y;
    if (dx * dx + dy * dy < radius * radius) {
      float dz = sqrtf(radius * radius - dx * dx - dy * dy);
      *n = dz / sqrtf(radius * radius);
      return dz + z;
    }
    return -INF;
  }
} Sphere;

// Kernel function for Ray Intersection
void kernel(int x, int y, Sphere* s, unsigned char* ptr) {
  int offset = x + y * DIM;
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
  int no_threads;
  int option;
  int x, y;
  unsigned char* bitmap;

  srand(time(NULL));

  if (argc != 3) {
    printf("> a.out [option] [filename.ppm]\n");
    printf(
        "for example, '> a.out 8 result.ppm' means executing OpenMP with 8 "
        "threads\n");
    exit(0);
  }

  FILE* fp = fopen(argv[2], "w");

  no_threads = atoi(argv[1]);

  // randomly generate spheres
  Sphere* temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
  for (int i = 0; i < SPHERES; i++) {
    temp_s[i].r = rnd(1.0f);
    temp_s[i].g = rnd(1.0f);
    temp_s[i].b = rnd(1.0f);
    temp_s[i].x = rnd(2000.0f) - 1000;
    temp_s[i].y = rnd(2000.0f) - 1000;
    temp_s[i].z = rnd(2000.0f) - 1000;
    temp_s[i].radius = rnd(200.0f) + 40;
  }
  bitmap = (unsigned char*)malloc(sizeof(unsigned char) * DIM * DIM * 4);

  // set number of threads
  omp_set_num_threads(no_threads);

  double start, end;

  // start timer
  start = omp_get_wtime();
  // set up schedule for OpenMP parallel for loop(Dynamic shcedule)
#pragma omp parallel for schedule(dynamic) private(x, y)
  for (x = 0; x < DIM; x++)
    for (y = 0; y < DIM; y++) kernel(x, y, temp_s, bitmap);
  // end timer
  end = omp_get_wtime();

  printf("With OpenMP: %f seconds\n", end - start);

  ppm_write(bitmap, DIM, DIM, fp);

  fclose(fp);
  free(bitmap);
  free(temp_s);

  return 0;
}