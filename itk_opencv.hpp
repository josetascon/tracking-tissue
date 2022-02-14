/*
* @Author: jose
* @Date:   2021-07-14 13:11:41
* @Last Modified by:   jose
* @Last Modified time: 2021-07-14 14:58:41
*/

// itk libs
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

// opencv libs
#include <opencv2/opencv.hpp>

template <typename type>
cv::Mat read_with_itk_to_opencv(std::string file_name)
{
    // Read image
    const int d = 2;
    using itkImageType = itk::Image<type, d>;
    typename itkImageType::Pointer image_itk = itkImageType::New();

    // const auto image_itk = itk::ReadImage<itkImageType>(file_name)
    using itkReaderType = itk::ImageFileReader<itkImageType>;
    typename itkReaderType::Pointer reader = itkReaderType::New();
    reader->SetFileName(file_name);
    reader->Update();
    image_itk = reader->GetOutput();

    type * data = image_itk->GetBufferPointer();

    // Copy to opencv
    int cols = image_itk->GetLargestPossibleRegion().GetSize()[0];
    int rows = image_itk->GetLargestPossibleRegion().GetSize()[1];
    int chs = 1;
    cv::Mat image_cv(rows, cols, CV_MAKETYPE(cv::DataType<type>::type, chs));
    memcpy(image_cv.data, data, rows*cols*chs * sizeof(type));
    return image_cv;
}

template <typename type>
void write_opencv_with_itk(cv::Mat & image_cv, std::string file_name)
{
    const int d = 2;
    using itkImageType = itk::Image<type, d>;
    typename itkImageType::Pointer image_itk = itkImageType::New();
    typename itkImageType::RegionType region;
    typename itkImageType::IndexType  start;
    typename itkImageType::SizeType itksize;
    
    start[0] = 0;
    start[1] = 0;
    itksize[0] = image_cv.cols();
    itksize[1] = image_cv.rows();
    region.SetSize(itksize);
    region.SetIndex(start);

    image_itk->SetRegions(region);
    image_itk->Allocate();

    int cols = image_cv.cols();
    int rows = image_cv.rows();
    int chs = 1;

    type * data = image_itk->GetBufferPointer();
    memcpy(data, image_cv.data, rows*cols*chs * sizeof(type));

    // itk::WriteImage(image_itk, file_name);
    using itkWriterType = itk::ImageFileWriter<itkImageType>;
    typename itkWriterType::Pointer writer = itkWriterType::New();
    writer->SetFileName(file_name);
    writer->SetInput(image_itk);
    writer->Update();

    return;
}
