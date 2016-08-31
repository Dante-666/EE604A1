#include <iostream>
#include <ctime>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

using namespace std;
using namespace boost;
using namespace cv;

int addGaussianNoise(Mat *src, Mat *dest, double mean, double variance);
void displayImage(Mat img);
int NL_mean(const Mat *src, Mat *dest, Size size, double h);

inline double gwf(const Mat input, Point p, Point q, Size size, double h);
