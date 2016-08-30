#include <iostream>
#include <ctime>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#define MU 0
#define SD sqrt(0.02)

int main(int argc, char** argv) {
    /**
     * Read the image in memory
     */
    if (argc < 2 || argc > 2) {
        std::cout<<"No filename was given...exiting..."<<std::endl;
        return -1;
    }

    cv::Mat img = cv::imread(argv[1], 0);
    /**
     * This is CV_8U type.
     * std::cout<<img.type()<<std::endl;
     */
    //cv::Mat noise = cv::Mat::zeros(img.size(), CV_8U);

    boost::mt19937 rng;// = new boost::mt19937();

    std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto seed = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    
    rng.seed(seed);

    boost::normal_distribution<double> kernel (MU, SD);
    //double a = SD;
    //std::cout<<a<<std::endl;
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<>> gauss(rng, kernel);

    double x = gauss();
    std::cout<<x<<std::endl;
    return 0;
}
