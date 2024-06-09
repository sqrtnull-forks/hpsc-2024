#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

__device__ inline double pow2(double x) { return x * x; }

__global__ void calc1(int nx, int ny, double dx, double dy, double dt,
                      double rho, float *u, float *v, float *b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < nx && j < ny) {
    int ji, jip, jim, jpi, jmi;
    ji = j * nx + i;
    jip = ji + 1;
    jim = ji - 1;
    jpi = ji + nx;
    jmi = ji - nx;

    if (j > 0 && j < ny - 1 && i > 0 && i < nx - 1) {
      b[ji] =
          rho *
          (1 / dt *
               ((u[jip] - u[jim]) / (2 * dx) + (v[jpi] - v[jmi]) / (2 * dy)) -
           pow2((u[jip] - u[jim]) / (2 * dx)) -
           2 * ((u[jpi] - u[jmi]) / (2 * dy) * (v[jip] - v[jim]) / (2 * dx)) -
           pow2(((v[jpi] - v[jmi]) / (2 * dy))));
    }
  }
}
__global__ void calc2(int nx, int ny, double dx, double dy, float *p, float *b,
                      float *pn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < nx && j < ny) {
    int ji, jip, jim, jpi, jmi;
    ji = j * nx + i;
    jip = ji + 1;
    jim = ji - 1;
    jpi = ji + nx;
    jmi = ji - nx;
    if (j > 0 && j < ny - 1 && i > 0 && i < nx - 1) {
      p[ji] = (pow2(dy) * (pn[jip] + pn[jim]) + pow2(dx) * (pn[jpi] + pn[jmi]) -
               b[ji] * pow2(dx) * pow2(dy)) /
              (2 * (pow2(dx) + pow2(dy)));
    }
  }
}
__global__ void calc31(int nx, int ny, float *p) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < ny) {
    p[j * nx] = p[j * nx + 1];
    p[(j + 1) * nx - 1] = p[(j + 1) * nx - 2];
  }
}
__global__ void calc32(int nx, int ny, float *p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nx) {
    p[i] = p[nx + i];
    p[(ny - 1) * nx + i] = 0;
  }
}
__global__ void calc4(int nx, int ny, double dx, double dy, double dt,
                      double rho, double nu, float *u, float *v, float *p,
                      float *un, float *vn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < nx && j < ny) {
    int ji, jip, jim, jpi, jmi;
    ji = j * nx + i;
    jip = ji + 1;
    jim = ji - 1;
    jpi = ji + nx;
    jmi = ji - nx;
    if (j > 0 && j < ny - 1 && i > 0 && i < nx - 1) {
      // Compute u[ji] and v[ji]
      u[ji] = un[ji] - un[ji] * dt / dx * (un[ji] - un[jim]) -
              un[ji] * dt / dy * (un[ji] - un[jmi]) -
              dt / (2 * rho * dx) * (p[jip] - p[jim]) +
              nu * dt / pow2(dx) * (un[jip] - 2 * un[ji] + un[jim]) +
              nu * dt / pow2(dy) * (un[jpi] - 2 * un[ji] + un[jmi]);
      v[ji] = vn[ji] - vn[ji] * dt / dx * (vn[ji] - vn[jim]) -
              vn[ji] * dt / dy * (vn[ji] - vn[jmi]) -
              dt / (2 * rho * dx) * (p[jpi] - p[jmi]) +
              nu * dt / pow2(dx) * (vn[jip] - 2 * vn[ji] + vn[jim]) +
              nu * dt / pow2(dy) * (vn[jpi] - 2 * vn[ji] + vn[jmi]);
    }
    if (i == 0) {
      u[j * nx] = 0;
      u[(j + 1) * nx - 1] = 0;
      v[j * nx] = 0;
      v[(j + 1) * nx - 1] = 0;
    }
    if (j == 0) {
      u[i] = 0;
      u[(ny - 1) * nx + i] = 1;
      v[i] = 0;
      v[(ny - 1) * nx + i] = 0;
    }
  }
}

using namespace std;
typedef vector<vector<float>> matrix;

int main() {
  int nx = 41;
  int ny = 41;
  int nt = 500;
  int nit = 50;
  double dx = 2. / (nx - 1);
  double dy = 2. / (ny - 1);
  double dt = .01;
  double rho = 1.;
  double nu = .02;

  float *u;
  float *v;
  float *p;
  float *b;
  float *un;
  float *vn;
  float *pn;
  int matrix_size = ny * nx * sizeof(float);
  cudaMallocManaged(&u, matrix_size);
  cudaMallocManaged(&v, matrix_size);
  cudaMallocManaged(&p, matrix_size);
  cudaMallocManaged(&b, matrix_size);
  cudaMallocManaged(&un, matrix_size);
  cudaMallocManaged(&vn, matrix_size);
  cudaMallocManaged(&pn, matrix_size);
  cudaMemset(u, 0, matrix_size);
  cudaMemset(v, 0, matrix_size);
  cudaMemset(p, 0, matrix_size);
  cudaMemset(b, 0, matrix_size);
  ofstream ufile("u.dat");
  ofstream vfile("v.dat");
  ofstream pfile("p.dat");
  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

  cudaDeviceSynchronize();
  for (int n = 0; n < nt; n++) {
    calc1<<<numBlocks, threadsPerBlock>>>(nx, ny, dx, dy, dt, rho, u, v, b);
    for (int it = 0; it < nit; it++) {
      cudaMemcpy(pn, p, matrix_size, cudaMemcpyDeviceToDevice);
      cudaDeviceSynchronize();
      calc2<<<numBlocks, threadsPerBlock>>>(nx, ny, dx, dy, p, b, pn);
      cudaDeviceSynchronize();
      calc31<<<numBlocks.y, threadsPerBlock.y>>>(nx, ny, p);
      cudaDeviceSynchronize();
      calc32<<<numBlocks.x, threadsPerBlock.x>>>(nx, ny, p);
      cudaDeviceSynchronize();
    }
    cudaMemcpy(un, u, matrix_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(vn, v, matrix_size, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    calc4<<<numBlocks, threadsPerBlock>>>(nx, ny, dx, dy, dt, rho, nu, u, v, p,
                                          un, vn);
    cudaDeviceSynchronize();
    if (n % 10 == 0) {
      for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
          ufile << u[j * nx + i] << " ";
      ufile << "\n";
      for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
          vfile << v[j * nx + i] << " ";
      vfile << "\n";
      for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
          pfile << p[j * nx + i] << " ";
      pfile << "\n";
    }
  }
  ufile.close();
  vfile.close();
  pfile.close();
}
