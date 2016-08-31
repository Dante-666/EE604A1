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

int addGaussianNoise(Mat *src, Mat *dest, double mean, double variance) {

    if (src->rows != dest->rows || src->cols != dest->cols) {
        cout<<"src and dest dimension mismatch..."<<endl;
        return -1;
    }
    
    mt19937 rng;

    chrono::time_point<std::chrono::system_clock> now = chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto seed = chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    
    rng.seed(seed);

    normal_distribution<double> kernel (mean, sqrt(variance));
    variate_generator<mt19937&, normal_distribution<>> gauss(rng, kernel);

    for (int i = 0; i < dest->rows; i++) {
        for(int j = 0; j < dest->cols; j++) {
            dest->at<float>(i, j) += static_cast<float>(gauss());
        }
    }

    return 0;
}

void displayImage(Mat img) {
    cout<<"Close the window to continue..."<<endl;

    namedWindow("Debug_Win", WINDOW_AUTOSIZE);
    imshow("Debug_Win", img);

    waitKey(0);
}

int main(int argc, char** argv) {
    /**
     * Read the image in memory
     */
    if (argc < 2 || argc > 2) {
        cout<<"No filename was given...exiting..."<<endl;
        return -1;
    }

    Mat img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

    Mat noisy = Mat::zeros(img.rows, img.cols, CV_32F);

    normalize(img, noisy, 0.f, 1.f, NORM_MINMAX, CV_32F);

    if (addGaussianNoise(&img, &noisy, 0.f, 0.02f)) return -1;

    displayImage(noisy);
    
    return 0;
}
