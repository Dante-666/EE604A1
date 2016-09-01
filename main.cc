#include "main.h"

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

int NL_mean(const Mat *src, Mat *dest, Size size, double h) {
    if (src->rows != dest->rows || src->cols != dest->cols) {
        cout<<"src and dest dimension mismatch..."<<endl;
        return -1;
    } else if (size.height != size.width) {
        cout<<"non square kernel, exiting..."<<endl;
        return -1;
    }

    /**
     * Pad the input array by mirroring.
     *
     */
    unsigned int pad = size.width/2;

    Mat extra = Mat::zeros(src->cols + 2 * pad,
                           src->rows + 2 * pad,
                           src->type());

    copyMakeBorder(*src, extra, pad, pad, pad, pad, BORDER_REFLECT_101);

    //cout<<extra.rows<<"--"<<extra.cols<<endl;
    //displayImage(extra);
    //
    
    //Point p(src->cols-1,src->rows-1);
    //Point q(src->cols-1,src->rows-1);

    //cout<<gwf(extra, p, q, size, 0.1)<<endl;
    //
    int n = 1;

    for (int i = 0; i < src->cols; i++) {
        for (int j = 0; j < src->rows; j++) {
            Point p(i, j);
            double norm_p = 0;
            double u_p = 0;

            for (int k = 0; k < src->cols; k++) {
                for (int l = 0; l < src->rows; l++) {
                    Point q(k, l);
                    double weight = gwf(extra, p, q, size, h);
                    norm_p += weight;
                    u_p += weight * src->at<double>(q);
                }
            }
            //cout<<p.x<<" -- "<<p.y<<endl;
            u_p /= norm_p;
            dest->at<float>(p) = u_p;
        
            if (float(i * src->cols + j)/float((src->cols-1)
                * (src->rows-1)) > n * 0.01)
                cout<<n++<<" percent complete"<<endl;
            //cout<<float(i * src->cols + j)/float((src->cols-1) * (src->rows-1))<<endl;
        }
    }

    return 0;
}

double gwf(const Mat input, Point p, Point q, Size size, double h) {

    Rect roi_p(p, size);
    Mat n_p = input(roi_p);

    Rect roi_q(q, size);
    Mat n_q = input(roi_q);

    /**
     * mean(Mat) results in scalar of dim 4, the first
     * of which is the first channel, which will be the
     * value of interest in grayscale.
     */
    
    return exp(-1 * pow(mean(n_p)[0] - mean(n_q)[0], 2)/pow(h, 2));
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
    Mat filt = Mat::zeros(img.rows, img.cols, CV_8U);

    img.convertTo(noisy, CV_32F, 1/255.f);

    if (addGaussianNoise(&img, &noisy, 0.f, 0.02f)) return -1;

    //displayImage(noisy);
    //Point p(49,37);
    //Point q(,3);

    //gwf(img, p, q, Size(5, 5), 0.1);

    NL_mean(&noisy, &filt, Size(5, 5), 0.1);
    
    vector<int> compression_params;
    compression_params.push_back(IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

    noisy.convertTo(filt, CV_8U, 255);

    //displayImage(filt);

    imwrite("noisy.png", filt, compression_params);

    return 0;
}
