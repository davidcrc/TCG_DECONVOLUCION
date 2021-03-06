// *******************************************************
// nvcc graficav2.cu `pkg-config --cflags --libs opencv`
// *******************************************************
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>

using namespace std;
using namespace cv;

void shift(Mat magI) {
 
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
 
    int cx = magI.cols/2;
    int cy = magI.rows/2;
 
    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
 
    Mat tmp;                            // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);                     // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

void show_espectrum(Mat complex )
{
 
    Mat magI;
    Mat planes[] = {Mat::zeros(complex.size(), CV_32F), Mat::zeros(complex.size(), CV_32F)};
    split(complex, planes);                // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))
 
    magnitude(planes[0], planes[1], magI);    // sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)
 
    // switch to logarithmic scale: log(1 + magnitude)
    magI += Scalar::all(1);
    log(magI, magI);
 
    shift(magI);
    normalize(magI, magI, 0, 1, NORM_MINMAX); 
                                              
    // invertir mascara
    if(true) {
        magI = Mat::ones(magI.size(), CV_32F) - magI;
    }
    imshow("spectrum", magI);
}

Mat resizeGPU(std::string input_file, std::string output_file, int w, int h)
{
	//
	Mat inputCpu = imread(input_file, CV_LOAD_IMAGE_COLOR);
	cuda::GpuMat input(inputCpu);
	if (input.empty())
	{
		std::cout << "Imagen no encontrada: " << input_file << std::endl;
		// return;
	}

	// salida
	cuda::GpuMat output;

	// cuda::resize(input, output, Size(1000,800), .25, 0.25, CV_INTER_LINEAR); // downscale 4x on both x and y
	cuda::resize(input, output, Size(w,h), 0.0, 0.0, CV_INTER_LINEAR); // downscale 4x on both x and y

	Mat outputCpu;
	output.download(outputCpu);
	imwrite(output_file, outputCpu);

	input.release();
    // output.release();
    return outputCpu;
}

cuda::GpuMat DFT(Mat img)
{
    img.convertTo(img, CV_32F, 1.0 / 255);

    std::vector<Mat> planes;
    planes.push_back(img);
    planes.push_back(Mat::zeros(img.size(), CV_32FC1));
    merge(planes, img);

    cuda::GpuMat imgGPU = cuda::GpuMat(img.size(), CV_32FC2);
    cuda::GpuMat imgSpectrum = cuda::GpuMat(img.size(), CV_32FC2);
    imgGPU.upload(img);
    
    cuda::Stream stream;
    cuda::dft(imgGPU, imgSpectrum, imgGPU.size(), 0, stream);
    imgGPU.release();
    imgGPU.release();
    return imgSpectrum;

}
void updateResult(Mat complex)
{
    Mat work;
    idft(complex, work);
//  dft(complex, work, DFT_INVERSE + DFT_SCALE);
    Mat planes[] = {Mat::zeros(complex.size(), CV_32F), Mat::zeros(complex.size(), CV_32F)};
    split(work, planes);                // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))
 
    magnitude(planes[0], planes[1], work);    // === sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)
    normalize(work, work, 0, 1, NORM_MINMAX);
    imshow("resultadoo", work);
}
Mat IDFT(cuda::GpuMat imgSpectrum)
{
    Mat imgFinal;

    cuda::GpuMat imgOut = cuda::GpuMat(imgSpectrum.size(), CV_32FC2);
    cuda::Stream stream;
    // cuda::dft(imgSpectrum, imgOut, imgGPU.size(), DFT_INVERSE, stream);
    cuda::dft(imgSpectrum, imgOut, imgSpectrum.size(), DFT_INVERSE, stream);

    vector<cuda::GpuMat> splitter(2);
    // cuda::GpuMat* splitter;
    cuda::split(imgOut, splitter, stream);
    stream.waitForCompletion();
    splitter[0].download(imgFinal);
    // imgOut.release();
    return imgFinal;
}
int main(int argc, char *argv[])
{

    if(argc!=2) {
        printf("Ha olvidado la imagen.\n"); 
        exit(1);
    }
    // resizeGPU("img.png","lena_b.png",1024,1024);

    
}
