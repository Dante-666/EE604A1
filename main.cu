#include "main.h"

__global__ void thread_gwf(double *d_C_p, double *d_u_p, double *d_mu, float *d_src, double h, unsigned long p, int rows, int cols) {
    /**
     * Exit the threads that are outside the image boundary.
     * Taken from my own code.
     * https://github.com/Dante-666/cuda-test/blob/master/src/ip/gaussblur.cu
     */
    if (blockIdx.x == (int) cols/blockDim.x && threadIdx.x + blockIdx.x * blockDim.x >= cols) return;
    else if (blockIdx.y == (int) rows/blockDim.y && threadIdx.y + blockIdx.y * blockDim.y >= rows) return;
    
    __shared__ double C_p;
    __shared__ double u_p;

    unsigned long toffset = threadIdx.x + threadIdx.y * cols;
    unsigned long boffset = blockIdx.y * blockDim.y * cols + blockDim.x * blockIdx.x;

    int q = toffset + boffset;
    double mu_p = d_mu[p];

    double w_pq = exp(-1 * pow((mu_p - d_mu[q])/h, 2));

    atomicAdd(&C_p, w_pq);
    atomicAdd(&u_p, d_src[q] * w_pq);

    __syncthreads();

    /**
     * Update the global memory per block.
     */
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(d_C_p, C_p);
        atomicAdd(d_u_p, u_p);
    }
}

int addGaussianNoise(Mat *src, Mat *dest, double mean, double variance) {

    if (src->rows != dest->rows || src->cols != dest->cols) {
        cout<<"src and dest dimension mismatch..."<<endl;
        return -1;
    }

    /**
     * Help via boost docs.
     * http://www.boost.org/doc/libs/1_61_0/doc/html/boost_random.html
     */
    
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

    Mat extra = Mat::zeros(src->rows + 2 * pad,
                           src->cols + 2 * pad,
                           src->type());
    Mat mu = Mat::zeros(src->rows, src->cols, CV_64F);

    copyMakeBorder(*src, extra, pad, pad, pad, pad, BORDER_REFLECT_101);

    int i, j;
    int n = 1;

    /**
     * Compute mean beforehand to avoid redundancy.
     */

    for (int i = 0; i < src->rows; i++) {
        for (int j = 0; j < src->cols; j++) {
            Point p(j, i);

            Rect roi_p(p, size); 
            Mat n_p = extra(roi_p);

            mu.at<double>(p) = mean(n_p)[0];
        }
    }

    cout<<"Mean matrix complete..."<<endl;

    /**
     * Allocate host and device memory for cuda.
     */

    double *h_C_p = new double;
    double *h_u_p = new double;
    double *d_C_p;
    double *d_u_p;

    double *d_mu;
    double *h_mu = new double[mu.rows * mu.cols];

    float *d_src;
    float *h_src = new float [src->rows * src->cols];

    int k = 0;

    for (i = 0; i < mu.rows; i++) {
        double *i_row = mu.ptr<double>(i);
        for (j = 0; j < mu.cols; j++) {
            h_mu[k++] = i_row[j];
        }
    }

    k = 0;

    for (i = 0; i < src->rows; i++) {
        const float *i_row = src->ptr<float>(i);
        for (j = 0; j < src->cols; j++) {
            h_src[k++] = i_row[j];
        }
    }

    cudaMalloc((void **) &d_C_p, sizeof(double));
    cudaMalloc((void **) &d_u_p, sizeof(double));

    cudaMalloc((void **) &d_mu, sizeof(double) * mu.rows * mu.cols);
    cudaMemcpy(d_mu, h_mu, sizeof(double) * mu.rows * mu.cols, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_src, sizeof(float) * src->rows * src->cols);
    cudaMemcpy(d_src, h_src, sizeof(float) * src->rows * src->cols, cudaMemcpyHostToDevice);

    dim3 grid(ceil(mu.cols/MAX_THREADS), ceil(mu.rows/MAX_THREADS), 1);
    dim3 block(MAX_THREADS, MAX_THREADS, 1);

    for (i = 0; i < src->rows; i++) {
        for (j = 0; j < src->cols; j++) {
            Point p(j, i);
            //TODO: Replace with CUDA Kerneli
            /**
            for (int k = 0; k < src->cols; k++) {
                for (int l = 0; l < src->rows; l++) {
                    Point q(k, l);
                    double weight = gwf(&mu, p, q, h);
                    norm_p += weight;
                    u_p += weight * src->at<float>(q);
                }
            }*/

            thread_gwf<<<grid, block>>>(d_C_p, d_u_p, d_mu, d_src, h, j + i * src->cols ,src->rows, src->cols);
            cudaMemcpy(h_C_p, d_C_p, sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_u_p, d_u_p, sizeof(double), cudaMemcpyDeviceToHost);

            dest->at<float>(p) = (*h_u_p)/(*h_C_p);
        
            if (float(i * src->rows + j)/float((src->cols-1)
                * (src->rows-1)) > n * .01f)
                cout<<n++<<" percent complete"<<endl;
        }
    }

    cudaFree(d_src);
    cudaFree(d_mu);
    cudaFree(d_C_p);
    cudaFree(d_u_p);

    delete h_C_p;
    delete h_u_p;

    delete h_mu;
    delete h_src;

    return 0;
}

double gwf(Mat *input, Point p, Point q, double h) {
    return exp(-1 * pow( ( input->at<double>(p) - input->at<double>(q))/h, 2 ) );
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

    cout<<img.rows<<"--"<<img.cols<<endl;

    Mat noisy = Mat::zeros(img.rows, img.cols, CV_32F);
    Mat filt = Mat::zeros(img.rows, img.cols, CV_32F);

    Mat disp = Mat::zeros(img.rows, img.cols, CV_8U);

    cout<<noisy.rows<<"--"<<noisy.cols<<endl;

    img.convertTo(noisy, CV_32F, 1/255.f);

    if (addGaussianNoise(&img, &noisy, 0.f, 0.02f)) return -1;

    NL_mean(&noisy, &filt, Size(5, 5), 0.1);
    
    vector<int> compression_params;
    compression_params.push_back(IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

    filt.convertTo(disp, CV_8U, 255);

    imwrite("noisy_cuda.png", disp, compression_params);

    return 0;
}
