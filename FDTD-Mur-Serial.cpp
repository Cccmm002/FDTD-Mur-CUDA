#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
using namespace std;

//#define CELL double**
typedef double **CELL;
#define NULL 0

#define PI 3.1415926535897
#define SQRT2 1.414213562373
double mu = 12.56637e-7;
double epsilon = 8.8542e-12;
double cv = 3e8;  //Velocity of light
double sigma = 0; //Electrical conductivity in vacuity

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

int total_time, length, cx, cy;
double dl, dt;
double *ezf; //Electrical field(Ez) (i,j,t)
double *hxf; //Magnetic field(Hx) (i,j+0.5,t)
double *hyf; //Magnetic field(Hx) (i+0.5,j,t)

CELL createCell(int a, int b)
{
	CELL res = NULL;
	res = (CELL)malloc(sizeof(double*)*a);
	for (int i = 0; i < a; i++)
	{
		res[i] = (double*)malloc(sizeof(double)*b);
		memset(res[i], 0, sizeof(double)*b);
	}
	return res;
}

void freeCell(CELL c, int a, int b)
{
	if (c != NULL)
	{
		for (int i = 0; i < a; i++)
			free(c[i]);
		free(c);
		c = NULL;
	}
}

int main()
{
	total_time = 200;
	length = 64;
	cx = 32; cy = 32;
	dl = 1e-6;
	double st = (1 / sqrt(2));
	dt = st*dl / cv;
	dt = dt / 2;

	cout << "Input the length of the field: ";
	cin >> length;
	cout << "Input time steps to be calculated: ";
	cin >> total_time;
	cout << "Input x and y coordinate of the wave source: ";
	cin >> cx >> cy;

	CELL ezf, ezfm1, ezfm2, hxf, hxfm1, hyf, hyfm1;
	ezf = createCell(length, length);
	hxf = createCell(length, length);
	hyf = createCell(length, length);
	ezfm1 = createCell(length, length);

	cout << "Calculation start..." << endl << "Time:" << endl;
	double t_start, t_end, duration;
	t_start = clock();

	//Coefficients
	double a = ca(dt); double b = cb(dt);
	double p = cp(dt); double q = cq(dt);
	//Main loop
	for (int t = 2; t < total_time; t++)
	{
		ezfm2 = ezfm1;
		ezfm1 = ezf;
		ezf = createCell(length, length);
		hxfm1 = hxf;
		hxf = createCell(length, length);
		hyfm1 = hyf;
		hyf = createCell(length, length);

		for (int i = 1; i < length - 1; i++)
		{
			for (int j = 1; j < length - 1; j++)
			{
				if (i == cx && j == cy) //Wave source
				{
					double frq = 1.5e13;
					ezf[i][j] = sin(t * dt * 2 * PI * frq);
				}
				else //Recursion
					ezf[i][j] = a*ezfm1[i][j] + b*((hyfm1[i][j] - hyfm1[i - 1][j]) / dl - (hxfm1[i][j] - hxfm1[i][j - 1]) / dl);
			}
		}

        //Coefficients for boundary conditions
		double coe1 = (cv*dt - dl) / (cv*dt + dl);
		double coe2 = (2 * dl) / (cv*dt + dl);
		double coe3 = (cv*cv*dt*dt) / (2 * dl*(cv*dt + dl));

		for (int j = 1; j < length - 1; j++)
		{
			ezf[0][j] = 0 - ezfm2[1][j] + coe1*(ezf[1][j] + ezfm2[0][j]) + coe2*(ezfm1[0][j] + ezfm1[1][j]) + coe3*(ezfm1[0][j + 1] - 2 * ezfm1[0][j] + ezfm1[0][j - 1] + ezfm1[1][j + 1] - 2 * ezfm1[1][j] + ezfm1[1][j - 1]);
			ezf[length - 1][j] = 0 - ezfm2[length - 2][j] + coe1*(ezf[length - 2][j] + ezfm2[length - 1][j]) + coe2*(ezfm1[length - 1][j] + ezfm1[length - 2][j]) + coe3*(ezfm1[length - 1][j + 1] - 2 * ezfm1[length - 1][j] + ezfm1[length - 1][j - 1] + ezfm1[length - 2][j + 1] - 2 * ezfm1[length - 2][j] + ezfm1[length - 2][j - 1]);
		}
		for (int i = 1; i < length - 1; i++)
		{
			ezf[i][0] = 0 - ezfm2[i][1] + coe1*(ezf[i][1] + ezfm2[i][0]) + coe2*(ezfm1[i][0] + ezfm1[i][1]) + coe3*(ezfm1[i + 1][0] - 2 * ezfm1[i][0] + ezfm1[i - 1][0] + ezfm1[i + 1][1] - 2 * ezfm1[i][1] + ezfm1[i - 1][1]);
			ezf[i][length - 1] = 0 - ezfm2[i][length - 2] + coe1*(ezf[i][length - 2] + ezfm2[i][length - 1]) + coe2*(ezfm1[i][length - 1] + ezfm1[i][length - 2]) + coe3*(ezfm1[i + 1][length - 1] - 2 * ezfm1[i][length - 1] + ezfm1[i - 1][length - 1] + ezfm1[i + 1][length - 2] - 2 * ezfm1[i][length - 2] + ezfm1[i - 1][length - 2]);
		}

		//Calculate corner of Ez
		double coe_cor = (cv*dt - SQRT2*dl) / (cv*dt + SQRT2*dl); //Coefficient of the formula
		ezf[0][0] = ezfm1[1][1] + coe_cor*(ezf[1][1] - ezfm1[0][0]);
		ezf[length - 1][0] = ezfm1[length - 2][1] + coe_cor*(ezf[length - 2][1] - ezfm1[length - 1][0]);
		ezf[0][length - 1] = ezfm1[1][length - 2] + coe_cor*(ezf[1][length - 2] - ezfm1[0][length - 1]);
		ezf[length - 1][length - 1] = ezfm1[length - 2][length - 2] + coe_cor*(ezf[length - 2][length - 2] - ezfm1[length - 1][length - 1]);

        //Ez -> Hx
		for (int i = 0; i < length; i++)
		{
			for (int j = 0; j < length - 1; j++)
				hxf[i][j] = p*hxfm1[i][j] - q*(ezf[i][j + 1] - ezf[i][j]) / dl;
		}
		for (int i = 0; i < length - 1; i++)
		{
			for (int j = 0; j < length; j++)
				hyf[i][j] = p*hyfm1[i][j] + q*(ezf[i + 1][j] - ezf[i][j]) / dl;
		}

		freeCell(ezfm2, length, length);
		freeCell(hxfm1, length, length);
		freeCell(hyfm1, length, length);

		if (t%100==0)
			cout << '\r' << t;
	}
	cout << endl;

	t_end = clock();
	duration = t_end - t_start;
	cout << "Time using: " << duration << " ms." << endl;

	for (int i = 0; i < length; i++)
	{
		for (int j = 0; j < length; j++)
			cout << ezf[i][j] << " ";
		cout << endl;
	}

    //Free
	freeCell(ezf, length, length);
	freeCell(ezfm1, length, length);
	freeCell(hxf, length, length);
	freeCell(hyf, length, length);

	return 0;
}
