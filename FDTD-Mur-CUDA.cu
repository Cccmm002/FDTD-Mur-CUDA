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

//ezf[i][j][t], hxf[i][j][t], hyf[i][j][t] are field to be calculated. (width, height) is the total size of the field.
//(s_x, s_y) is the point of source. 't' is the current time. 'dl' is the size of cell. 'dt' is time interval.
//'a' and 'b' are the coeffiecients of the media.
__global__ void calcEz(double *ezf, double *hxf, double *hyf, int width, int height, int s_x, int s_y, int t, double dl, double dt, double a, double b)
{
	int tx = threadIdx.x; int ty = threadIdx.y;
	int thx = blockIdx.x*blockDim.x + tx;
	int thy = blockIdx.y*blockDim.y + ty;
	if (thx < 0 || thy < 0 || thx >= (width - 1) || thy >= (height - 1)) return;

	//All index of elements of H add 1, and 0 is the index for halo elements
	__shared__ double hx[BLOCK_LENGTH + 1][BLOCK_LENGTH + 1];
	__shared__ double hy[BLOCK_LENGTH + 1][BLOCK_LENGTH + 1];
	hx[tx + 1][ty + 1] = hxf[(t - 1)*width*height + width*thy + thx];
	hy[tx + 1][ty + 1] = hyf[(t - 1)*width*height + width*thy + thx];

	//Copy halo elements
	if (tx == 0 && thx != 0) hy[0][ty + 1] = hyf[(t - 1)*width*height + width*thy + thx - 1];
	if (ty == 0 && thy != 0) hx[tx + 1][0] = hxf[(t - 1)*width*height + width*(thy - 1) + thx];

	__syncthreads();

	if (thx == 0 || thy == 0) return;

	if (thx == s_x && thy == s_y)
	{
		//Source
		double frq = 1.5e13;
		ezf[t*width*height + width*thy + thx] = sin(t * dt * 2 * PI * frq);
	}
	else //Recursion
		ezf[t*width*height + width*thy + thx] = a*ezf[(t - 1)*width*height + width*thy + thx] + b*((hy[tx + 1][ty + 1] - hy[tx][ty + 1]) / dl - (hx[tx + 1][ty + 1] - hx[tx + 1][ty]) / dl);
}

__global__ void calcH(double *ezf, double *hxf, double *hyf, int width, int height, int t, double dl, double p, double q)
{
	int tx = threadIdx.x; int ty = threadIdx.y;
	int thx = blockIdx.x*blockDim.x + tx;
	int thy = blockIdx.y*blockDim.y + ty;
	if (thx < 0 || thy < 0 || thx >= (width - 1) || thy >= (height - 1)) return;

	__shared__ double ez[BLOCK_LENGTH + 1][BLOCK_LENGTH + 1];
	if (tx == 0 && ty == 0)
		ez[0][0] = ezf[t*width*height + width*thy + thx];
	ez[tx + 1][ty] = ezf[t*width*height + width*thy + thx + 1];
	ez[tx][ty + 1] = ezf[t*width*height + width*(thy + 1) + thx];

	__syncthreads();

	hxf[t*width*height + width*thy + thx] = p*hxf[(t - 1)*width*height + width*thy + thx] - q*(ez[tx][ty + 1] - ez[tx][ty]) / dl;
	hyf[t*width*height + width*thy + thx] = p*hyf[(t - 1)*width*height + width*thy + thx] + q*(ez[tx + 1][ty] - ez[tx][ty]) / dl;
}

__global__ void calcLeftBoundary(double *ezf, int width, int height, int t, double coe1, double coe2, double coe3)
{
	int tx = threadIdx.x;
	int thx = blockIdx.x*blockDim.x + tx;

	__shared__ double left[BLOCK_LENGTH + 2];
	__shared__ double inner[BLOCK_LENGTH + 2];
	left[tx + 1] = ezf[(t - 1)*height*width + width*thx];
	inner[tx + 1] = ezf[(t - 1)*height*width + width*thx + 1];
	if (tx == 0 && thx != 0)
	{
		left[0] = ezf[(t - 1)*height*width + width*(thx - 1)];
		inner[0] = ezf[(t - 1)*height*width + width*(thx - 1) + 1];
	}
	if (tx == BLOCK_LENGTH - 1 && thx != height - 1)
	{
		left[BLOCK_LENGTH + 1] = ezf[(t - 1)*height*width + width*(thx + 1)];
		inner[BLOCK_LENGTH + 1] = ezf[(t - 1)*height*width + width*(thx + 1) + 1];
	}

	__syncthreads();

	if (!(thx == 0 || thx == height - 1))
		ezf[t*width*height + width*thx] = 0 - ezf[(t - 2)*width*height + width*thx + 1] + coe1*(ezf[t*width*height + width*thx + 1] + ezf[(t - 2)*width*height + width*thx]) + coe2*(left[tx + 1] + inner[tx + 1]) + coe3*(left[tx + 2] - 2 * left[tx + 1] + left[tx] + inner[tx + 2] - 2 * inner[tx + 1] + inner[tx]);
}

__global__ void calcRightBoundary(double *ezf, int width, int height, int t, double coe1, double coe2, double coe3)
{
	int tx = threadIdx.x;
	int thx = blockIdx.x*blockDim.x + tx;

	__shared__ double right[BLOCK_LENGTH + 2];
	__shared__ double inner[BLOCK_LENGTH + 2];
	right[tx + 1] = ezf[(t - 1)*height*width + width*(thx + 1) - 1];
	inner[tx + 1] = ezf[(t - 1)*height*width + width*(thx + 1) - 2];
	if (tx == 0 && thx != 0)
	{
		right[0] = ezf[(t - 1)*height*width + width*thx - 1];
		inner[0] = ezf[(t - 1)*height*width + width*thx - 2];
	}
	if (tx == BLOCK_LENGTH - 1 && thx != height - 1)
	{
		right[BLOCK_LENGTH + 1] = ezf[(t - 1)*height*width + width*(thx + 2) - 1];
		inner[BLOCK_LENGTH + 1] = ezf[(t - 1)*height*width + width*(thx + 2) - 2];
	}

	__syncthreads();

	if (!(thx == 0 || thx == height - 1))
		ezf[t*width*height + width*thx + width - 1] = 0 - ezf[(t - 2)*height*width + width*(thx + 1) - 2] + coe1*(ezf[t*height*width + width*(thx + 1) - 2] + ezf[(t - 2)*height*width + width*(thx + 1) - 1]) + coe2*(right[tx + 1] + inner[tx + 1]) + coe3*(right[tx + 2] - 2 * right[tx + 1] + right[tx] + inner[tx + 2] - 2 * inner[tx + 1] + inner[tx]);
}

__global__ void calcDownBoundary(double *ezf, int width, int height, int t, double coe1, double coe2, double coe3)
{
	int tx = threadIdx.x;
	int thx = blockIdx.x*blockDim.x + tx;

	__shared__ double down[BLOCK_LENGTH + 2];
	__shared__ double inner[BLOCK_LENGTH + 2];
	down[tx + 1] = ezf[(t - 1)*height*width + thx];
	inner[tx + 1] = ezf[(t - 1)*height*width + width + thx];
	if (tx == 0 && thx != 0)
	{
		down[0] = ezf[(t - 1)*height*width + thx - 1];
		inner[0] = ezf[(t - 1)*height*width + width + thx - 1];
	}
	if (tx == BLOCK_LENGTH - 1 && thx != width - 1)
	{
		down[BLOCK_LENGTH + 1] = ezf[(t - 1)*height*width + thx + 1];
		inner[BLOCK_LENGTH + 1] = ezf[(t - 1)*height*width + width + thx + 1];
	}

	__syncthreads();

	if (!(thx == 0 || thx == width - 1))
		ezf[t*width*height + thx] = 0 - ezf[(t - 2)*width*height + width + thx] + coe1*(ezf[t*width*height + width + thx] + ezf[(t - 2)*width*height + thx]) + coe2*(down[tx + 1] + inner[tx + 1]) + coe3*(down[tx + 2] - 2 * down[tx + 1] + down[tx] + inner[tx + 2] - 2 * inner[tx + 1] + inner[tx]);
}

__global__ void calcUpBoundary(double *ezf, int width, int height, int t, double coe1, double coe2, double coe3)
{
	int tx = threadIdx.x;
	int thx = blockIdx.x*blockDim.x + tx;

	__shared__ double up[BLOCK_LENGTH + 2];
	__shared__ double inner[BLOCK_LENGTH + 2];
	up[tx + 1] = ezf[(t - 1)*height*width + width*(height - 1) + thx];
	inner[tx + 1] = ezf[(t - 1)*height*width + width*(height - 2) + thx];
	if (tx == 0 && thx != 0)
	{
		up[0] = ezf[(t - 1)*height*width + width*(height - 1) + thx - 1];
		inner[0] = ezf[(t - 1)*height*width + width*(height - 2) + thx - 1];
	}
	if (tx == BLOCK_LENGTH - 1 && thx != width - 1)
	{
		up[BLOCK_LENGTH + 1] = ezf[(t - 1)*height*width + width*(height - 1) + thx + 1];
		inner[BLOCK_LENGTH + 1] = ezf[(t - 1)*height*width + width*(height - 2) + thx + 1];
	}

	__syncthreads();

	if (!(thx == 0 || thx == width - 1))
		ezf[t*width*height + width*(height - 1) + thx] = 0 - ezf[(t - 2)*width*height + width*(height - 2) + thx] + coe1*(ezf[t*width*height + width*(height - 2) + thx] + ezf[(t - 2)*width*height + width*(height - 1) + thx]) + coe2*(up[tx + 1] + inner[tx + 1]) + coe3*(up[tx + 2] - 2 * up[tx + 1] + up[tx] + inner[tx + 2] - 2 * inner[tx + 1] + inner[tx]);
}

__global__ void calcCorner(double *ezf, int width, int height, int t, double coe)
{
	int thx = blockIdx.x*blockDim.x + threadIdx.x;
	if (thx == 0)      //left-down
		ezf[t*width*height] = ezf[(t - 1)*width*height + width + 1] + coe*(ezf[t*width*height + width + 1] - ezf[(t - 1)*width*height]);
	else if (thx == 1) //right-down
		ezf[t*width*height + width - 1] = ezf[(t - 1)*width*height + width * 2 - 2] + coe*(ezf[t*width*height + width * 2 - 2] - ezf[(t - 1)*width*height + width - 1]);
	else if (thx == 2) //left-up
		ezf[t*width*height + width*(height - 1)] = ezf[(t - 1)*width*height + width*(height - 2) + 1] + coe*(ezf[t*width*height + width*(height - 2) + 1] - ezf[(t - 1)*width*height + width*(height - 1)]);
	else            //right-up
		ezf[(t + 1)*width*height - 1] = ezf[t*width*height - width - 2] + coe*(ezf[(t + 1)*width*height - width - 2] - ezf[t*width*height - 1]);
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

	int size = sizeof(double)*width*height*time;
	double *d_ezf; double *d_hxf; double *d_hyf;
	cudaMalloc((void**)&d_ezf, size);
	cudaMalloc((void**)&d_hxf, size);
	cudaMalloc((void**)&d_hyf, size);
	cudaMemset(d_ezf, 0, size);
	cudaMemset(d_hxf, 0, size);
	cudaMemset(d_hyf, 0, size);

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

	for (int t = 2; t < time; t++)
	{
		calcEz<<<DimGrid, DimBlock >>>(d_ezf, d_hxf, d_hyf, width, height, cx, cy, t, dl, dt, a, b);

		//Boundary conditions
		calcLeftBoundary<<<dg_h, db_h >>>(d_ezf, width, height, t, coe1, coe2, coe3);
		calcRightBoundary<<<dg_h, db_h >>>(d_ezf, width, height, t, coe1, coe2, coe3);
		calcDownBoundary<<<dg_w, db_w >>>(d_ezf, width, height, t, coe1, coe2, coe3);
		calcUpBoundary<<<dg_w, db_w >>>(d_ezf, width, height, t, coe1, coe2, coe3);
		//Corner conditions
		calcCorner<<<dg_cor, db_cor >>>(d_ezf, width, height, t, coe_cor);

		calcH<<<DimGrid, DimBlock >>>(d_ezf, d_hxf, d_hyf, width, height, t, dl, p, q);
	}

	double *output = (double*)malloc(size);
	cudaMemcpy(output, d_ezf, size, cudaMemcpyDeviceToHost);

	t_end = clock();
	duration = t_end - t_start;
	cout << "Time using: " << duration << " ms." << endl << endl;

	for (int t = 0; t < time; t++)
	{
		cout << "Time " << t << ":" << endl;
		cout << "Ez:" << endl;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
				cout << output[t*width*height + i*width + j] << " ";
			cout << endl;
		}
		cout << endl;
	}

	free(output);

	cudaFree(d_ezf); cudaFree(d_hxf); cudaFree(d_hyf);

	return 0;
}
