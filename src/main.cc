#include "main.h"
#include "parallel.h"

float *h_src;
float *h_dest;
float *h_mean;

int addGaussianNoise(Mat *dest, double mean, double variance) {
    
    mt19937 rng;

    chrono::time_point<std::chrono::system_clock> now = chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto seed = chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    
    rng.seed(seed);

    normal_distribution<double> kernel (mean, sqrt(variance));
    variate_generator<mt19937&, normal_distribution<>> gauss(rng, kernel);

    for (int i = 0; i < dest->rows; i++) {
        for(int j = 0; j < dest->cols; j++) {
            Point p(j, i);
            dest->at<float>(p) += static_cast<float>(gauss());
        }
    }

    return 0;
}

double snr(const Mat *src, const Mat *dest) {
    double sq_dist_err = 0.f;
    double sq_obs = 0.f;

    for (int i = 0; i < src->rows; i++) {
        for (int j = 0; j < src->cols; j++) {
            Point p(j, i);

            float diff = src->at<float>(p) - dest->at<float>(p);

            sq_dist_err += pow(diff, 2);
            sq_obs += pow(src->at<float>(p), 2);
        }
    }

    return 10 * log10(sq_obs/sq_dist_err);
}

void medianFilter(const Mat *src, Mat *dest, Size size) {
    Mat extra;

    unsigned int pad = size.width/2;
    
    padMirror(src, &extra, pad);

    for (int i = 0; i < src->rows; i++) {
        for (int j = 0; j < src->cols; j++) {
            Point p(j, i);

            Rect roi_p(p, size); 
            Mat n_p = extra(roi_p);

	    dest->at<float>(p) = median(n_p, dest->at<float>(p));
        }
    }
}

float median(const Mat n_p, float val) {
    vector<float> roi;
   
    for (int i = 0; i < n_p.rows; i++) {
        for (int j = 0; j < n_p.cols; j++) {
            Point p(j, i);

	    roi.push_back(n_p.at<float>(p));
        }
    }

    int n = n_p.rows * n_p.cols/2 ;

    range::nth_element(roi, roi.begin() + n);

    int i = 0;

    vector<float>::iterator it = roi.begin();

    while (i++ < n) {
        it++;
    }

    if (exp(-1 * pow((*it - val)/0.1f, 2) < 0.02f)) return *it;
    else return val;

}

void displayImage(Mat img) {
    cout<<"Close the window to continue..."<<endl;

    namedWindow("Debug_Win", WINDOW_AUTOSIZE);
    imshow("Debug_Win", img);

    waitKey(0);
}

void padMirror(const Mat *src, Mat *extra, unsigned int pad) {
    int top = pad;
    Mat topRows;

    while (top > 0) topRows.push_back(src->row(top--));
    
    topRows.push_back(src->rowRange(0, src->rows));

    unsigned int bottom = 0; 
    Mat botRows;
    while (bottom < pad) botRows.push_back(src->row(src->rows - 2 - bottom++));

    topRows.push_back(botRows);

    int left = pad;
    Mat leftCols = topRows.col(left--);

    while (left > 0) hconcat(leftCols, topRows.col(left--), leftCols);

    hconcat(leftCols, topRows.colRange(0, src->cols), leftCols);

    unsigned int right = 0;
    Mat rightCols = topRows.col(topRows.cols - 2 - right++);

    while (right < pad) hconcat(rightCols, topRows.col(topRows.cols - 2 - right++), rightCols);

    hconcat(leftCols, rightCols, leftCols);

    *extra = leftCols.clone();

}

int init_host_data(const Mat *src, Size size) {
    if (size.height != size.width) {
        cout<<"non square kernel, exiting..."<<endl;
        return -1;
    }
    
    unsigned long length = src->rows * src->cols;
    h_src = (float *) malloc(sizeof(float) * length);
    h_dest = (float *) malloc(sizeof(float) * length);
    h_mean = (float *) malloc(sizeof(float) * length);

    /**
     * Pad the input array by mirroring.
     *
     */
    unsigned int pad = size.width/2;

    Mat extra;

    padMirror(src, &extra, pad);

    int i, j, l, m;
    int k = 0;

    Mat kern = getGaussianKernel(size.width, 0.1, CV_32F);

    /**
     * Compute mean beforehand to avoid redundancy.
     */

    for (i = 0; i < src->rows; i++) {
        for (j = 0; j < src->cols; j++) {
            Point p(j, i);

            Rect roi_p(p, size); 
            Mat n_p = extra(roi_p);
            double sum = 0;

            for (l = 0; l < kern.rows; l++) {
                for (m = 0; m < kern.cols; m++) {
                    Point q(m, l);
                    sum += n_p.at<float>(q) * kern.at<float>(q);
                } 
            }

            h_mean[k] = float(sum);
	    h_src[k++] = src->at<float>(p);
        }
    }

    return 0;
}

void clear_host_data(Mat *dest) {
    free(h_src);
    free(h_mean);

    int k = 0;
    float *p;

    /**
     * Copy the array back to cv::Mat
     */
    for (int i = 0; i < dest->rows; i++) {
        p = dest->ptr<float>(i);
        for (int j = 0; j < dest->cols; j++) {
            p[j] = h_dest[k++];
        }
    }

    free(h_dest);
}

void help() {
    cout<<"This program computes NL means using a CUDA device."<<endl
        <<"A device is absolutely necessary in order to run this program."<<endl
        <<"The code is supposed to be run like this."<<endl<<endl;
    
    cout<<"\"run <filename> <n>\""<<endl
        <<"Where <n> = [1, 2, 3, 4, 5] for stepwise execution."<<endl
        <<"1. Adds gaussian noise(0,0.02f) to the image and writes it out."<<endl
        <<"2. Computes the NL Mean using 5x5 window and writes the result out."<<endl
        <<"3. Repeats 2. for window sizes 7x7 and 11x11 and writes the result out."<<endl
        <<"4. Computes NL Mean for different noise powers = [0.02f, 0.05f, 0.1f, 0.2f]"<<endl
        <<"5. Repeats 4. for aforementioned noise powers."<<endl<<endl;

    cout<<"Image files names are self explanatory"<<endl
        <<"<filter_type>_<window>_<noise_var>_<step>.png"<<endl<<endl;

    cout<<"snr.log is generated and appended with the step information"<<endl
        <<"and snr and runtime values"<<endl<<endl;

    cout<<"Libraries and documentation used :"<<endl
        <<"1. OpenCV 3.1.0"<<endl
        <<"2. Boost 1.60.0"<<endl
        <<"3. CUDA 7.5.18"<<endl
        <<"4. std c++11"<<endl<<endl;

    cout<<"run  Copyright (C) 2016  sjs"<<endl;
    cout<<"This program is free software: you can redistribute it and/or modify"<<endl
        <<"it under the terms of the GNU General Public License as published by"<<endl
        <<"the Free Software Foundation, either version 3 of the License, or"<<endl
        <<"(at your option) any later version."<<endl<<endl
        <<"This program is distributed in the hope that it will be useful,"<<endl
        <<"but WITHOUT ANY WARRANTY; without even the implied warranty of"<<endl
        <<"MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the"<<endl
        <<"GNU General Public License for more details."<<endl<<endl
        <<"You should have received a copy of the GNU General Public License"<<endl
        <<"along with this program.  If not, see <http://www.gnu.org/licenses/>."<<endl;
}

int main(int argc, char** argv) {
    /**
     * Read the image in memory
     */
    if (argc < 3 || argc > 3) {
        cout<<"Proper arguments were not given...exiting..."<<endl;
        help();
        return -1;
    }

    ofstream f("snr.log", ofstream::app);
    time_t now = chrono::system_clock::to_time_t(chrono::system_clock::now());
    f<<endl<<"Timestamp : "<<ctime(&now)<<endl;

    Mat img = imread(argv[1], IMREAD_GRAYSCALE);

    cout<<img.rows<<"--"<<img.cols<<endl;

    Mat img_f = Mat::zeros(img.rows, img.cols, CV_32F);
    Mat noisy = Mat::zeros(img.rows, img.cols, CV_32F);
    Mat filt = Mat::zeros(img.rows, img.cols, CV_32F);
    Mat filt_m = Mat::zeros(img.rows, img.cols, CV_32F);

    float time;

    Mat disp = Mat::zeros(img.rows, img.cols, CV_8U);

    vector<int> compression_params;
    compression_params.push_back(IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

    img.convertTo(img_f, CV_32F, 1/255.f);
    
    if (strncmp(argv[2], "1", 1) == 0) {
        /**
         * Add gaussian noise N(0, 0.02f)
         */
        img.convertTo(noisy, CV_32F, 1/255.f);

        if (addGaussianNoise(&noisy, 0.f, 0.02f)) return -1;
    
        f<<"1. Noisy Image with sigma = 0.02f : "<<snr(&img_f, &noisy)<<endl;

        noisy.convertTo(disp, CV_8U, 255);
        imwrite("noisy_0.02_1.png", disp, compression_params);

        Mat noisy_u8 = Mat::zeros(img.rows, img.cols, CV_8U);
        Mat filt_u8 = Mat::zeros(img.rows, img.cols, CV_8U);

        img.convertTo(noisy, CV_32F, 1/255.f);
        
        addGaussianNoise(&noisy, 0.f, 0.02f);

        noisy.convertTo(noisy_u8, CV_8U, 255);

        fastNlMeansDenoising(noisy_u8, filt_u8, 3, 7, 35);

        imwrite("test.png", filt_u8, compression_params);
    }
    else if (strncmp(argv[2], "2", 1) == 0) {

        /**
         * Add gaussian noise N(0, 0.02f)
         */
        img.convertTo(noisy, CV_32F, 1/255.f);
        
        if (addGaussianNoise(&noisy, 0.f, 0.02f)) return -1;
    
        f<<"2. Noisy Image with sigma = 0.02f : "<<snr(&img_f, &noisy)<<endl;
        
        /**
         * Filter image with 5x5 window.
         */

        init_host_data(&noisy, Size(5, 5));
        NL_mean(h_src, h_mean, h_dest, 0.12, noisy.rows, noisy.cols, &time);
        clear_host_data(&filt);

        f<<"2. Time taken to filter by GPU : "<<boost::format("%1.3f") % time<<" milliseconds"<<endl;
        f<<"2. NLM Filtered Image [5x5] with noise sigma = 0.02f : "<<snr(&img_f, &filt)<<endl;
    
        filt.convertTo(disp, CV_8U, 255);
        imwrite("filt_5x5_0.02_2.png", disp, compression_params);

        medianFilter(&filt, &filt_m, Size(3, 3));

        f <<"2. Median NLM Filtered Image [5x5] with noise sigma = 0.02f : "<<snr(&img_f, &filt_m)<<endl;

        filt_m.convertTo(disp, CV_8U, 255);
        imwrite("mfilt_5x5_0.02_2.png", disp, compression_params);
    }
    else if(strncmp(argv[2], "3", 1) == 0) {

        /**
         * Add gaussian noise N(0, 0.02f)
         */
        img.convertTo(noisy, CV_32F, 1/255.f);

        if (addGaussianNoise(&noisy, 0.f, 0.02f)) return -1;
    
        f<<"3. Noisy Image with sigma = 0.02f : "<<snr(&img_f, &noisy)<<endl;

        /**
         * Repeat for N(0, 0.02f) for 7x7 window.
         */

        init_host_data(&noisy, Size(7, 7));
        NL_mean(h_src, h_mean, h_dest, 0.12, noisy.rows, noisy.cols, &time);
        clear_host_data(&filt);

        f <<"3. Time taken to filter by GPU : "<<boost::format("%1.3f") % time<<" milliseconds"<<endl;
        f <<"3. NLM Filtered Image [7x7] with sigma = 0.02f : "<<snr(&img_f, &filt)<<endl;
    
        filt.convertTo(disp, CV_8U, 255);
        imwrite("filt_7x7_0.02_3.png", disp, compression_params);

        medianFilter(&filt, &filt_m, Size(3,3));

        f <<"3. Median NLM Filtered Image [7x7] with sigma = 0.02f : "<<snr(&img_f, &filt_m)<<endl;

        filt_m.convertTo(disp, CV_8U, 255);
        imwrite("mfilt_7x7_0.02_3.png", disp, compression_params);

        /**
         * Repeat from N(0, 0.02f) for 11x11 window
         */

        init_host_data(&noisy, Size(11, 11));
        NL_mean(h_src, h_mean, h_dest, 0.12, noisy.rows, noisy.cols, &time);
        clear_host_data(&filt);

        f <<"3. Time taken to filter by GPU : "<<boost::format("%1.3f") % time<<" milliseconds"<<endl;
        f <<"3. NLM Filtered Image [11x11] with sigma = 0.02f : "<<snr(&img_f, &filt)<<endl;
    
        filt.convertTo(disp, CV_8U, 255);
        imwrite("filt_11x11_0.02_3.png", disp, compression_params);

        medianFilter(&filt, &filt_m, Size(3,3));

        f <<"3. Median NLM Filtered Image [11x11] with sigma = 0.02f : "<<snr(&img_f, &filt_m)<<endl;

        filt_m.convertTo(disp, CV_8U, 255);
        imwrite("mfilt_11x11_0.02_3.png", disp, compression_params);
    }
    else if(strncmp(argv[2], "4", 1) == 0) {
    
        /**
         * Add N(0, 0.02) noise now.
         */
        float var = 0.02f;
        string var_s = "0.02";
        img.convertTo(noisy, CV_32F, 1/255.f);

        if (addGaussianNoise(&noisy, 0.f, var)) return -1;
    
        f <<"4. Noisy Image with sigma = "<<var_s<<" : "<<snr(&img_f, &noisy)<<endl;

        Size custom(5, 5);
        string filt_fname = "filt_";
        string noise_fname = "noise_";
        string fname;
        
        /**
         * Repeat from N(0, 0.02f) for custom window
         */
    
        init_host_data(&noisy, custom);
        NL_mean(h_src, h_mean, h_dest, 0.12, noisy.rows, noisy.cols, &time);
        clear_host_data(&filt);

        f <<"4. Time taken to filter by GPU : "<<boost::format("%1.3f") % time<<" milliseconds"<<endl;
        f <<"4. NLM Filtered Image ["<<custom.height<<"x"<<custom.width<<"] with var = "<<var<<" : "<<snr(&img_f, &filt)<<endl;
    
        filt.convertTo(disp, CV_8U, 255);
        fname = filt_fname + to_string(custom.width) + "x" + to_string(custom.height) + "_" + var_s + "_4.png";
        imwrite(fname, disp, compression_params);

        medianFilter(&filt, &filt_m, Size(3,3));

        f <<"4. Median NLM Filtered Image ["<<custom.height<<"x"<<custom.width<<"] with var = "<<var<<" : "<<snr(&img_f, &filt_m)<<endl;

        filt_m.convertTo(disp, CV_8U, 255);
        fname = "m" + filt_fname + to_string(custom.width) + "x" + to_string(custom.height) + "_" + var_s + "_4.png";
        imwrite(fname, disp, compression_params);

        /**
         * Add N(0, 0.05) noise now.
         */
        var = 0.05f;
        var_s = "0.05";
        img.convertTo(noisy, CV_32F, 1/255.f);

        if (addGaussianNoise(&noisy, 0.f, var)) return -1;
    
        f <<"4. Noisy Image with sigma = "<<var_s<<" : "<<snr(&img_f, &noisy)<<endl;
    
        noisy.convertTo(disp, CV_8U, 255);
        fname = noise_fname + var_s + "_4.png"; 
        imwrite(fname, disp, compression_params);

        /**
         * Repeat from N(0, 0.05f) for custom window
         */
    
        init_host_data(&noisy, custom);
        NL_mean(h_src, h_mean, h_dest, 0.12, noisy.rows, noisy.cols, &time);
        clear_host_data(&filt);

        f <<"4. Time taken to filter by GPU : "<<boost::format("%1.3f") % time<<" milliseconds"<<endl;
        f <<"4. NLM Filtered Image ["<<custom.height<<"x"<<custom.width<<"] with var = "<<var<<" : "<<snr(&img_f, &filt)<<endl;
    
        filt.convertTo(disp, CV_8U, 255);
        fname = filt_fname + to_string(custom.width) + "x" + to_string(custom.height) + "_" + var_s + "_4.png";
        imwrite(fname, disp, compression_params);

        medianFilter(&filt, &filt_m, Size(3,3));

        f <<"4. Median NLM Filtered Image ["<<custom.height<<"x"<<custom.width<<"] with var = "<<var<<" : "<<snr(&img_f, &filt_m)<<endl;

        filt_m.convertTo(disp, CV_8U, 255);
        fname = "m" + filt_fname + to_string(custom.width) + "x" + to_string(custom.height) + "_" + var_s + "_4.png";
        imwrite(fname, disp, compression_params);

        /**
         * Add N(0, 0.1) noise now.
         */
        var = 0.1f;
        var_s = "0.1";
        img.convertTo(noisy, CV_32F, 1/255.f);

        if (addGaussianNoise(&noisy, 0.f, var)) return -1;
    
        f <<"4. Noisy Image with sigma = "<<var_s<<" : "<<snr(&img_f, &noisy)<<endl;
    
        noisy.convertTo(disp, CV_8U, 255);
        fname = noise_fname + var_s + "_4.png"; 
        imwrite(fname, disp, compression_params);

        /**
         * Repeat from N(0, 0.1f) for custom window
         */
    
        init_host_data(&noisy, custom);
        NL_mean(h_src, h_mean, h_dest, 0.12, noisy.rows, noisy.cols, &time);
        clear_host_data(&filt);

        f <<"4. Time taken to filter by GPU : "<<boost::format("%1.3f") % time<<" milliseconds"<<endl;
        f <<"4. NLM Filtered Image ["<<custom.height<<"x"<<custom.width<<"] with var = "<<var<<" : "<<snr(&img_f, &filt)<<endl;
    
        filt.convertTo(disp, CV_8U, 255);
        fname = filt_fname + to_string(custom.width) + "x" + to_string(custom.height) + "_" + var_s + "_4.png";
        imwrite(fname, disp, compression_params);

        medianFilter(&filt, &filt_m, Size(3,3));

        f <<"4. Median NLM Filtered Image ["<<custom.height<<"x"<<custom.width<<"] with var = "<<var<<" : "<<snr(&img_f, &filt_m)<<endl;

        filt_m.convertTo(disp, CV_8U, 255);
        fname = "m" + filt_fname + to_string(custom.width) + "x" + to_string(custom.height) + "_" + var_s + "_4.png";
        imwrite(fname, disp, compression_params);

        /**
         * Add N(0, 0.2) noise now.
         */
        var = 0.2f;
        var_s = "0.2";
        img.convertTo(noisy, CV_32F, 1/255.f);

        if (addGaussianNoise(&noisy, 0.f, var)) return -1;
    
        f <<"4. Noisy Image with sigma = "<<var_s<<" : "<<snr(&img_f, &noisy)<<endl;
    
        noisy.convertTo(disp, CV_8U, 255);
        fname = noise_fname + var_s + "_4.png"; 
        imwrite(fname, disp, compression_params);
    
        /**
         * Repeat from N(0, 0.2f) for custom window
         */
        init_host_data(&noisy, Size(5, 5));
        NL_mean(h_src, h_mean, h_dest, 0.12, noisy.rows, noisy.cols, &time);
        clear_host_data(&filt);

        f <<"4. Time taken to filter by GPU : "<<boost::format("%1.3f") % time<<" milliseconds"<<endl;
        f <<"4. NLM Filtered Image ["<<custom.height<<"x"<<custom.width<<"] with var = "<<var<<" : "<<snr(&img_f, &filt)<<endl;
    
        filt.convertTo(disp, CV_8U, 255);
        fname = filt_fname + to_string(custom.width) + "x" + to_string(custom.height) + "_" + var_s + "_4.png";
        imwrite(fname, disp, compression_params);

        medianFilter(&filt, &filt_m, Size(3,3));

        f <<"4. Median NLM Filtered Image ["<<custom.height<<"x"<<custom.width<<"] with var = "<<var<<" : "<<snr(&img_f, &filt_m)<<endl;

        filt_m.convertTo(disp, CV_8U, 255);
        fname = "m" + filt_fname + to_string(custom.width) + "x" + to_string(custom.height) + "_" + var_s + "_4.png";
        imwrite(fname, disp, compression_params);

    }
    else if(strncmp(argv[2], "5", 1) == 0) {

        Mat med_filt = Mat::zeros(img.rows, img.cols, CV_8U);
        Mat med_noisy = Mat::zeros(img.rows, img.cols, CV_8U);
        
        /**
         * Add N(0, 0.02f)
         */
        img.convertTo(noisy, CV_32F, 1/255.f);

        if (addGaussianNoise(&noisy, 0.f, 0.02f)) return -1;
    
        f <<"5. Noisy Image with sigma = 0.02f : "<<snr(&img_f, &noisy)<<endl;

        blur(noisy, filt, Size(11, 11), Point(-1,-1));
            
        f <<"5. AVG Filtered Image [11x11] with noise sigma = 0.02f : "<<snr(&img_f, &filt)<<endl;
    
        filt.convertTo(disp, CV_8U, 255);
        imwrite("avg_filt_11x11_0.02_5.png", disp, compression_params);

        GaussianBlur(noisy, filt, Size(11, 11), 0.f, 0.f, BORDER_REFLECT_101);

        f <<"5. Gaussian Filtered Image [11x11] with noise sigma = 0.02f : "<<snr(&img_f, &filt)<<endl;

        filt.convertTo(disp, CV_8U, 255);
        imwrite("gauss_filt_11x11_0.02_5.png", disp, compression_params);

        noisy.convertTo(med_noisy, CV_8U, 255);
        medianBlur(med_noisy, med_filt, 11);

        med_filt.convertTo(filt, CV_32F, 1/255.f);
        f <<"5. Median Filtered Image [11x11] with noise sigma = 0.02f : "<<snr(&img_f, &filt)<<endl;

        imwrite("median_filt_11x11_0.02_5.png", med_filt, compression_params);

        /**
         * Add N(0, 0.05) noise now.
         */
        img.convertTo(noisy, CV_32F, 1/255.f);

        if (addGaussianNoise(&noisy, 0.f, 0.05)) return -1;
    
        f <<"5. Noisy Image with sigma = 0.05f : "<<snr(&img_f, &noisy)<<endl;
        
        blur(noisy, filt, Size(11, 11), Point(-1,-1));
            
        f <<"5. AVG Filtered Image [11x11] with noise sigma = 0.05f : "<<snr(&img_f, &filt)<<endl;
    
        filt.convertTo(disp, CV_8U, 255);
        imwrite("avg_filt_11x11_0.05_5.png", disp, compression_params);

        GaussianBlur(noisy, filt, Size(11, 11), 0.f, 0.f, BORDER_REFLECT_101);

        f <<"5. Gaussian Filtered Image [11x11] with noise sigma = 0.05f : "<<snr(&img_f, &filt)<<endl;

        filt.convertTo(disp, CV_8U, 255);
        imwrite("gauss_filt_11x11_0.05_5.png", disp, compression_params);

        noisy.convertTo(med_noisy, CV_8U, 255);
        medianBlur(med_noisy, med_filt, 11);

        med_filt.convertTo(filt, CV_32F, 1/255.f);
        f <<"5. Median Filtered Image [11x11] with noise sigma = 0.05f : "<<snr(&img_f, &filt)<<endl;

        imwrite("median_filt_11x11_0.05_5.png", med_filt, compression_params);

        /**
         * Add N(0, 0.1) noise now.
         */

        img.convertTo(noisy, CV_32F, 1/255.f);

        if (addGaussianNoise(&noisy, 0.f, 0.1)) return -1;
    
        f <<"5. Noisy Image with sigma = 0.1f : "<<snr(&img_f, &noisy)<<endl;
        
        blur(noisy, filt, Size(11, 11), Point(-1,-1));
            
        f <<"5. AVG Filtered Image [11x11] with noise sigma = 0.1f : "<<snr(&img_f, &filt)<<endl;
    
        filt.convertTo(disp, CV_8U, 255);
        imwrite("avg_filt_11x11_0.1_5.png", disp, compression_params);

        GaussianBlur(noisy, filt, Size(11, 11), 0.f, 0.f, BORDER_REFLECT_101);

        f <<"5. Gaussian Filtered Image [11x11] with noise sigma = 0.1f : "<<snr(&img_f, &filt)<<endl;

        filt.convertTo(disp, CV_8U, 255);
        imwrite("gauss_filt_11x11_0.1_5.png", disp, compression_params);

        noisy.convertTo(med_noisy, CV_8U, 255);
        medianBlur(med_noisy, med_filt, 11);

        med_filt.convertTo(filt, CV_32F, 1/255.f);
        f <<"5. Median Filtered Image [11x11] with noise sigma = 0.1f : "<<snr(&img_f, &filt)<<endl;

        imwrite("median_filt_11x11_0.1_5.png", med_filt, compression_params);

        /**
         * Add N(0, 0.2) noise now.
         */
    
        img.convertTo(noisy, CV_32F, 1/255.f);

        if (addGaussianNoise(&noisy, 0.f, 0.2)) return -1;
    
        f <<"5. Noisy Image with sigma = 0.2f : "<<snr(&img_f, &noisy)<<endl;
        
        blur(noisy, filt, Size(11, 11), Point(-1,-1));
            
        f <<"5. AVG Filtered Image [11x11] with noise sigma = 0.2f : "<<snr(&img_f, &filt)<<endl;
    
        filt.convertTo(disp, CV_8U, 255);
        imwrite("avg_filt_11x11_0.2_5.png", disp, compression_params);

        GaussianBlur(noisy, filt, Size(11, 11), 0.f, 0.f, BORDER_REFLECT_101);

        f <<"5. Gaussian Filtered Image [11x11] with noise sigma = 0.2f : "<<snr(&img_f, &filt)<<endl;

        filt.convertTo(disp, CV_8U, 255);
        imwrite("gauss_filt_11x11_0.2_5.png", disp, compression_params);

        noisy.convertTo(med_noisy, CV_8U, 255);
        medianBlur(med_noisy, med_filt, 11);

        med_filt.convertTo(filt, CV_32F, 1/255.f);
        f <<"5. Median Filtered Image [11x11] with noise sigma = 0.2f : "<<snr(&img_f, &filt)<<endl;

        filt.convertTo(disp, CV_8U, 255);
        imwrite("median_filt_11x11_0.2_5.png", med_filt, compression_params);
    } else {
        help();
    }
    
    f.close();

    return 0;
}
