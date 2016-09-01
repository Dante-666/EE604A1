#include <iostream>
#include <ctime>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#define MAX_THREADS 32

using namespace std;
using namespace boost;
using namespace cv;

__device__ double atomicAdd(double* a, double b) { return b; }

__global__ void thread_gwf(double *d_C_p, double *d_u_p, double *d_mu, float *d_src, double h, unsigned long p, int rows, int cols);

int addGaussianNoise(Mat *src, Mat *dest, double mean, double variance);
void displayImage(Mat img);
int NL_mean(const Mat *src, Mat *dest, Size size, double h);

inline double gwf(Mat *mu, Point p, Point q, double h);
