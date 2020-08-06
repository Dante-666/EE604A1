#ifndef __MAIN_H__
#define __MAIN_H__

#include <iostream>
#include <fstream>
#include <ctime>
#include <chrono>

#include <string.h>

#include <opencv2/core/core.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/format.hpp>

using namespace std;
using namespace boost;
using namespace cv;
/**
 * Adds noise drawn from N(mean, variance) to each pixel in dest.
 */
int addGaussianNoise(Mat *dest, double mean, double variance);
/**
 * Computes SNR by using the formula,
 * 10 log (sum(square actual)/sum(squared dist)
 */
double snr(const Mat *src, const Mat *dest);
/**
 * Tries to get the best estimate for pixel content val, in it's
 * neighbourhood n_p.
 */
float median(const Mat n_p, const float val);
/**
 * Displays the image.
 */
void displayImage(Mat img);

/**
 * Pads the image by mirroring rows and columns.
 */
void padMirror(const Mat *src, Mat *extra, unsigned int pad);

/**
 * Loads 2-D cv:Mat objects into serial 1-D arrays for parallel
 * Processing. Also computes the mean in n_p according to size.
 */
int init_host_data(const Mat *src, Size size);
/**
 * Clears memory used in the last iteration and copies the results
 * back to dest.
 */
void clear_host_data(Mat *dest);

/**
 * Prints help and license information.
 */
void help();

#endif
