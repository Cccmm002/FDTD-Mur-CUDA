//////////////////////////////////////////////////////////////
//  2D FDTD solution for Mur's Absorbing Boundary Condition
//  Using GPU acceleration (CUDA implementation)
//       Simple harmonic excitation source
//////////////////////////////////////////////////////////////
//  Set your code generation to "compute_20,sm_20"
//  when you compile this code.
//////////////////////////////////////////////////////////////

#include <iostream>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
using namespace std;

#define BLOCK_LENGTH 32

#define PI 3.1415926535897
#define SQRT2 1.414213562373
double mu = 12.56637e-7;
double epsilon = 8.8542e-12;
double cv = 3e8;  //Velocity of light
double sigma = 0; //Electrical conductivity in vacuity

//Coefficients for media
inline double ca(double t)
{
	double tmp = sigma*t / (2 * epsilon);
	return (1 - tmp) / (1 + tmp);
}
inline double cb(double t)
{
	double tmp = t / epsilon;
	return tmp / (1 + tmp*(sigma / 2));
}
inline double cp(double t)
{
	double tmp = sigma*t / (2 * mu);
	return (1 - tmp) / (1 + tmp);
}
inline double cq(double t)
{
	double tmp = t / mu;
	return tmp / (1 + tmp*(sigma / 2));
}

//ezf[i][j] is the electrical field to be calculated, ezfm1[i][j] is electrical field in the previous time step;
//hxfm1[i][j] and hyfm1[i][j] are the magnetic field in the previous time step;
//(width, height) is the total size of the field;
//(s_x, s_y) is the point of source. 't' is the current time. 'dl' is the size of cell. 'dt' is time interval;
//'a' and 'b' are the coeffiecients of the media.
__global__ void calcEz(double *ezf, double *ezfm1, double *hxfm1, double *hyfm1, int width, int height, int s_x, int s_y, int t, double dl, double dt, double a, double b)
{
	int tx = threadIdx.x; int ty = threadIdx.y;
	int thx = blockIdx.x*blockDim.x + tx;
	int thy = blockIdx.y*blockDim.y + ty;
	if (thx < 0 || thy < 0 || thx >= (width - 1) || thy >= (height - 1)) return;

	//All index of elements of H add 1, and 0 is the index for halo elements
	__shared__ double hx[BLOCK_LENGTH + 1][BLOCK_LENGTH + 1];
	__shared__ double hy[BLOCK_LENGTH + 1][BLOCK_LENGTH + 1];
	hx[tx + 1][ty + 1] = hxfm1[width*thy + thx];
	hy[tx + 1][ty + 1] = hyfm1[width*thy + thx];

	//Copy halo elements
	if (tx == 0 && thx != 0) hy[0][ty + 1] = hyfm1[width*thy + thx - 1];
	if (ty == 0 && thy != 0) hx[tx + 1][0] = hxfm1[width*(thy - 1) + thx];

	__syncthreads();

	if (thx == 0 || thy == 0) return;

	if (thx == s_x && thy == s_y)
	{
		//Source
		double frq = 1.5e13;
		ezf[width*thy + thx] = sin(t * dt * 2 * PI * frq);
	}
	else //Recursion
		ezf[width*thy + thx] = a*ezfm1[width*thy + thx] + b*((hy[tx + 1][ty + 1] - hy[tx][ty + 1]) / dl - (hx[tx + 1][ty + 1] - hx[tx + 1][ty]) / dl);
}

__global__ void calcH(double *ezf, double *hxf, double *hxfm1, double *hyf, double *hyfm1, int width, int height, double dl, double p, double q)
{
	int tx = threadIdx.x; int ty = threadIdx.y;
	int thx = blockIdx.x*blockDim.x + tx;
	int thy = blockIdx.y*blockDim.y + ty;
	if (thx < 0 || thy < 0 || thx >= (width - 1) || thy >= (height - 1)) return;

	__shared__ double ez[BLOCK_LENGTH + 1][BLOCK_LENGTH + 1];
	if (tx == 0 && ty == 0)
		ez[0][0] = ezf[width*thy + thx];
	ez[tx + 1][ty] = ezf[width*thy + thx + 1];
	ez[tx][ty + 1] = ezf[width*(thy + 1) + thx];

	__syncthreads();

	hxf[width*thy + thx] = p*hxfm1[width*thy + thx] - q*(ez[tx][ty + 1] - ez[tx][ty]) / dl;
	hyf[width*thy + thx] = p*hyfm1[width*thy + thx] + q*(ez[tx + 1][ty] - ez[tx][ty]) / dl;
}

__global__ void calcLeftBoundary(double *ezf, double *ezfm1, double *ezfm2, int width, int height, double coe1, double coe2, double coe3)
{
	int tx = threadIdx.x;
	int thx = blockIdx.x*blockDim.x + tx;

	__shared__ double left[BLOCK_LENGTH + 2];
	__shared__ double inner[BLOCK_LENGTH + 2];
	left[tx + 1] = ezfm1[width*thx];
	inner[tx + 1] = ezfm1[width*thx + 1];
	if (tx == 0 && thx != 0)
	{
		left[0] = ezfm1[width*(thx - 1)];
		inner[0] = ezfm1[width*(thx - 1) + 1];
	}
	if (tx == BLOCK_LENGTH - 1 && thx != height - 1)
	{
		left[BLOCK_LENGTH + 1] = ezfm1[width*(thx + 1)];
		inner[BLOCK_LENGTH + 1] = ezfm1[width*(thx + 1) + 1];
	}

	__syncthreads();

	if (!(thx == 0 || thx == height - 1))
		ezf[width*thx] = 0 - ezfm2[width*thx + 1] + coe1*(ezf[width*thx + 1] + ezfm2[width*thx]) + coe2*(left[tx + 1] + inner[tx + 1]) + coe3*(left[tx + 2] - 2 * left[tx + 1] + left[tx] + inner[tx + 2] - 2 * inner[tx + 1] + inner[tx]);
}

__global__ void calcRightBoundary(double *ezf, double *ezfm1, double *ezfm2, int width, int height, double coe1, double coe2, double coe3)
{
	int tx = threadIdx.x;
	int thx = blockIdx.x*blockDim.x + tx;

	__shared__ double right[BLOCK_LENGTH + 2];
	__shared__ double inner[BLOCK_LENGTH + 2];
	right[tx + 1] = ezfm1[width*(thx + 1) - 1];
	inner[tx + 1] = ezfm1[width*(thx + 1) - 2];
	if (tx == 0 && thx != 0)
	{
		right[0] = ezfm1[width*thx - 1];
		inner[0] = ezfm1[width*thx - 2];
	}
	if (tx == BLOCK_LENGTH - 1 && thx != height - 1)
	{
		right[BLOCK_LENGTH + 1] = ezfm1[width*(thx + 2) - 1];
		inner[BLOCK_LENGTH + 1] = ezfm1[width*(thx + 2) - 2];
	}

	__syncthreads();

	if (!(thx == 0 || thx == height - 1))
		ezf[width*thx + width - 1] = 0 - ezfm2[width*(thx + 1) - 2] + coe1*(ezf[width*(thx + 1) - 2] + ezfm2[width*(thx + 1) - 1]) + coe2*(right[tx + 1] + inner[tx + 1]) + coe3*(right[tx + 2] - 2 * right[tx + 1] + right[tx] + inner[tx + 2] - 2 * inner[tx + 1] + inner[tx]);
}

__global__ void calcDownBoundary(double *ezf, double *ezfm1, double *ezfm2, int width, int height, double coe1, double coe2, double coe3)
{
	int tx = threadIdx.x;
	int thx = blockIdx.x*blockDim.x + tx;

	__shared__ double down[BLOCK_LENGTH + 2];
	__shared__ double inner[BLOCK_LENGTH + 2];
	down[tx + 1] = ezfm1[thx];
	inner[tx + 1] = ezfm1[width + thx];
	if (tx == 0 && thx != 0)
	{
		down[0] = ezfm1[thx - 1];
		inner[0] = ezfm1[width + thx - 1];
	}
	if (tx == BLOCK_LENGTH - 1 && thx != width - 1)
	{
		down[BLOCK_LENGTH + 1] = ezfm1[thx + 1];
		inner[BLOCK_LENGTH + 1] = ezfm1[width + thx + 1];
	}

	__syncthreads();

	if (!(thx == 0 || thx == width - 1))
		ezf[thx] = 0 - ezfm2[width + thx] + coe1*(ezf[width + thx] + ezfm2[thx]) + coe2*(down[tx + 1] + inner[tx + 1]) + coe3*(down[tx + 2] - 2 * down[tx + 1] + down[tx] + inner[tx + 2] - 2 * inner[tx + 1] + inner[tx]);
}

__global__ void calcUpBoundary(double *ezf, double *ezfm1, double *ezfm2, int width, int height, double coe1, double coe2, double coe3)
{
	int tx = threadIdx.x;
	int thx = blockIdx.x*blockDim.x + tx;

	__shared__ double up[BLOCK_LENGTH + 2];
	__shared__ double inner[BLOCK_LENGTH + 2];
	up[tx + 1] = ezfm1[width*(height - 1) + thx];
	inner[tx + 1] = ezfm1[width*(height - 2) + thx];
	if (tx == 0 && thx != 0)
	{
		up[0] = ezfm1[width*(height - 1) + thx - 1];
		inner[0] = ezfm1[width*(height - 2) + thx - 1];
	}
	if (tx == BLOCK_LENGTH - 1 && thx != width - 1)
	{
		up[BLOCK_LENGTH + 1] = ezfm1[width*(height - 1) + thx + 1];
		inner[BLOCK_LENGTH + 1] = ezfm1[width*(height - 2) + thx + 1];
	}

	__syncthreads();

	if (!(thx == 0 || thx == width - 1))
		ezf[width*(height - 1) + thx] = 0 - ezfm2[width*(height - 2) + thx] + coe1*(ezf[width*(height - 2) + thx] + ezfm2[width*(height - 1) + thx]) + coe2*(up[tx + 1] + inner[tx + 1]) + coe3*(up[tx + 2] - 2 * up[tx + 1] + up[tx] + inner[tx + 2] - 2 * inner[tx + 1] + inner[tx]);
}

__global__ void calcCorner(double *ezf, double *ezfm1, int width, int height, double coe)
{
	int thx = blockIdx.x*blockDim.x + threadIdx.x;
	if (thx == 0)      //left-down
		ezf[0] = ezfm1[width + 1] + coe*(ezf[width + 1] - ezfm1[0]);
	else if (thx == 1) //right-down
		ezf[width - 1] = ezfm1[width * 2 - 2] + coe*(ezf[width * 2 - 2] - ezfm1[width - 1]);
	else if (thx == 2) //left-up
		ezf[width*(height - 1)] = ezfm1[width*(height - 2) + 1] + coe*(ezf[width*(height - 2) + 1] - ezfm1[width*(height - 1)]);
	else            //right-up
		ezf[width*height - 1] = ezfm1[width*height - width - 2] + coe*(ezf[width*height - width - 2] - ezfm1[width*height - 1]);
}

int main()
{
	int width = 128;
	int height = 128;
	int time = 400;
	int cx = 32; int cy = 32;
	double dl = 1e-6;
	double st = 1 / SQRT2;
	double dt = st*dl / cv;
	dt = dt / 2;

	cout << "Input width and height of the field: ";
	cin >> width >> height;
	cout << "Input time steps to be calculated: ";
	cin >> time;
	cout << "Input x and y coordinate of the wave source: ";
	cin >> cx >> cy;

	int size = sizeof(double)*width*height;
	double *ezf, *ezfm1, *ezfm2, *hxf, *hxfm1, *hyf, *hyfm1;
	cudaMalloc((void**)&ezf, size); cudaMemset(ezf, 0, size);
	cudaMalloc((void**)&ezfm1, size); cudaMemset(ezfm1, 0, size);
	cudaMalloc((void**)&hxf, size); cudaMemset(hxf, 0, size);
	cudaMalloc((void**)&hyf, size); cudaMemset(hyf, 0, size);

	//Coefficients of media
	double a = ca(dt); double b = cb(dt);
	double p = cp(dt); double q = cq(dt);
	//Coefficients for boundary conditions
	double coe1 = (cv*dt - dl) / (cv*dt + dl);
	double coe2 = (2 * dl) / (cv*dt + dl);
	double coe3 = (cv*cv*dt*dt) / (2 * dl*(cv*dt + dl));
	//Coefficient for the corner
	double coe_cor = (cv*dt - SQRT2*dl) / (cv*dt + SQRT2*dl);

	double t_start, t_end, duration;
	t_start = clock();

	dim3 DimBlock(BLOCK_LENGTH, BLOCK_LENGTH, 1);
	dim3 DimGrid((width - 1) / BLOCK_LENGTH + 1, (height - 1) / BLOCK_LENGTH + 1, 1);

	dim3 db_h(BLOCK_LENGTH, 1, 1);
	dim3 dg_h((height - 1) / BLOCK_LENGTH + 1, 1, 1);
	dim3 db_w(BLOCK_LENGTH, 1, 1);
	dim3 dg_w((width - 1) / BLOCK_LENGTH + 1, 1, 1);
	dim3 db_cor(4, 1, 1); dim3 dg_cor(1, 1, 1);

	for (int t = 0; t<time; t++)
	{
		ezfm2 = ezfm1;
		ezfm1 = ezf;
		cudaMalloc((void**)&ezf, size); cudaMemset(ezf, 0, size);
		hxfm1 = hxf;
		cudaMalloc((void**)&hxf, size); cudaMemset(hxf, 0, size);
		hyfm1 = hyf;
		cudaMalloc((void**)&hyf, size); cudaMemset(hyf, 0, size);

		calcEz<<<DimGrid, DimBlock >>>(ezf, ezfm1, hxfm1, hyfm1, width, height, cx, cy, t, dl, dt, a, b);

		//Boundary conditions
		calcLeftBoundary<<<dg_h, db_h >>>(ezf, ezfm1, ezfm2, width, height, coe1, coe2, coe3);
		calcRightBoundary<<<dg_h, db_h >>>(ezf, ezfm1, ezfm2, width, height, coe1, coe2, coe3);
		calcDownBoundary<<<dg_w, db_w >>>(ezf, ezfm1, ezfm2, width, height, coe1, coe2, coe3);
		calcUpBoundary<<<dg_w, db_w >>>(ezf, ezfm1, ezfm2, width, height, coe1, coe2, coe3);
		//Corner conditions
		calcCorner<<<dg_cor, db_cor >>>(ezf, ezfm1, width, height, coe_cor);

		calcH <<<DimGrid, DimBlock >>>(ezf, hxf, hxfm1, hyf, hyfm1, width, height, dl, p, q);

		//Output -------
		//You can add code here to copy data from array 'ezf', 'hxf' and 'hyf' in device to host in order to output.

		cudaFree(ezfm2); cudaFree(hxfm1); cudaFree(hyfm1);
	}

	double *output = (double*)malloc(size);
	cudaMemcpy(output, ezf, size, cudaMemcpyDeviceToHost);

	t_end = clock();
	duration = t_end - t_start;
	cout << "Time using: " << duration << " ms." << endl << endl;

	for (int i = 0; i<height; i++)
	{
		for (int j = 0; j<width; j++)
			cout << output[i*width + j] << " ";
		cout << endl;
	}

	free(output);
	cudaFree(ezf); cudaFree(hxf); cudaFree(hyf);
	cudaFree(ezfm1);

	return 0;
}
