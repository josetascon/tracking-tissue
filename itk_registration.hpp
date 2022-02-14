/*
* @Author: jose
* @Date:   2021-09-14 19:41:47
* @Last Modified by:   jose
* @Last Modified time: 2021-09-15 12:01:40
*/

// std libs
#include <iostream>

// itk libs
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

#include "itkImageRegistrationMethodv4.h"
#include "itkLBFGSOptimizerv4.h"
#include "itkLBFGS2Optimizerv4.h"

#include <itkTransform.h>
#include "itkBSplineTransform.h"
#include "itkBSplineTransformInitializer.h"
#include "itkBSplineTransformParametersAdaptor.h"

#include <itkMeanSquaresImageToImageMetricv4.h>
#include <itkCorrelationImageToImageMetricv4.h>
#include <itkANTSNeighborhoodCorrelationImageToImageMetricv4.h>
#include <itkMattesMutualInformationImageToImageMetricv4.h>

#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"


// boost libs
#include <boost/program_options.hpp>

namespace po = boost::program_options;
namespace fs = std::filesystem;

// using namespace itk;
namespace itk
{

template <typename TRegistration>
class RegistrationInterfaceCommand : public itk::Command
{
public:
    using Self = RegistrationInterfaceCommand;
    using Superclass = itk::Command;
    using Pointer = itk::SmartPointer<Self>;
    itkNewMacro(Self);
 
protected:
    RegistrationInterfaceCommand() = default;

public:
    using RegistrationType = TRegistration;
    using RegistrationPointer = RegistrationType *;
    using OptimizerType = itk::LBFGSOptimizerv4;
    using OptimizerPointer = OptimizerType *;

    void Execute(itk::Object * object, const itk::EventObject & event) override
    {
        if (!(itk::MultiResolutionIterationEvent().CheckEvent(&event)))
        {
          return;
        }

        auto registration = static_cast<RegistrationPointer>(object);
        auto optimizer = static_cast<OptimizerPointer>(registration->GetModifiableOptimizer());
        
        unsigned int currentLevel = registration->GetCurrentLevel();
        typename RegistrationType::ShrinkFactorsPerDimensionContainerType
            shrinkFactors = registration->GetShrinkFactorsPerDimension(currentLevel);
        typename RegistrationType::SmoothingSigmasArrayType 
            smoothingSigmas = registration->GetSmoothingSigmasPerLevel();

        std::cout << "-------------------------------------" << std::endl;
        std::cout << " Current level = " << currentLevel << std::endl;
        std::cout << "    shrink factor = " << shrinkFactors << std::endl;
        std::cout << "    smoothing sigma = " << smoothingSigmas[currentLevel] << std::endl;
        std::cout << std::endl;

        std::vector<int> iters = registration->get_level_iterations();
        optimizer->SetNumberOfIterations(iters[currentLevel]);
    };

    void Execute(const itk::Object *, const itk::EventObject & event) override
    {
        return;
    };
};

class CommandIterationUpdate : public itk::Command
{
public:
    using Self = CommandIterationUpdate;
    using Superclass = itk::Command;
    using Pointer = itk::SmartPointer<Self>;
    itkNewMacro(Self);

protected:
    CommandIterationUpdate() = default;

private:
    unsigned int m_PreviousIteration{ 0 };

public:
    using OptimizerType = itk::LBFGSOptimizerv4;
    // using OptimizerType = itk::LBFGS2Optimizerv4;
    using OptimizerPointer = const OptimizerType *;

    void Execute(itk::Object *caller, const itk::EventObject & event) override
    {
        Execute( (const itk::Object *) caller, event);
    };

    void Execute(const itk::Object * object, const itk::EventObject & event) override
    {
        OptimizerPointer optimizer = static_cast< OptimizerPointer >( object );
        if( !(itk::IterationEvent().CheckEvent( &event )) or (m_PreviousIteration == optimizer->GetCurrentIteration()) )
        {
            return;
        }

        printf("%d \t %f \n", optimizer->GetCurrentIteration(), optimizer->GetValue());
        m_PreviousIteration = optimizer->GetCurrentIteration();
    };
};

template<typename TFixedImage,
    typename TMovingImage,
    typename TOutputTransform = Transform<double, TFixedImage::ImageDimension, TFixedImage::ImageDimension>,
    typename TVirtualImage = TFixedImage,
    typename TPointSet = PointSet<unsigned int, TFixedImage::ImageDimension> >
class ITK_TEMPLATE_EXPORT RegistrationBsplines : public itk::ImageRegistrationMethodv4<TFixedImage,TMovingImage,TOutputTransform,TVirtualImage,TPointSet>
{
public:
    using Self = RegistrationBsplines;
    using Superclass = itk::ImageRegistrationMethodv4<TFixedImage,TMovingImage,TOutputTransform,TVirtualImage,TPointSet>;
    using Pointer = itk::SmartPointer<Self>;
    itkNewMacro(Self);

protected:
    
    // unsigned int SplineOrder = 3;
    typename itk::ImageToImageMetricv4< TFixedImage, TMovingImage>::Pointer similarity;

    bool verbose;
    std::vector<int> level_scales;
    std::vector<int> level_sigmas;
    std::vector<int> level_iterations;

public:
    using PixelType = float;
    using CoordinatesType = double;
    // using ImageType = itk::Image<PixelType, Dimension>;
    // using InterpolatorType  = itk::LinearInterpolateImageFunction< ImageType, PixelType >;

    using RegistrationType  = itk::ImageRegistrationMethodv4< TFixedImage, TMovingImage >;

public:

    std::vector<int> get_level_iterations() { return level_iterations; };

    void set_verbose(bool vv){ verbose = vv; };

    void set_levels(std::vector<int> iter, std::vector<int> scale, std::vector<int> sigma)
    {
        assert((iter.size() == scale.size()) && (sigma.size() == scale.size()) );
        level_iterations = iter;
        level_scales = scale;
        level_sigmas = sigma;
    };

    void optimizer()
    {
        using OptimizerType = itk::LBFGSOptimizerv4;
        // using OptimizerType = itk::LBFGS2Optimizerv4;
        typename OptimizerType::Pointer optimizer = OptimizerType::New();
        this->SetOptimizer(optimizer);

        // Scales
        // using ScalesEstimatorType = itk::RegistrationParameterScalesFromPhysicalShift<MetricType>;
        // typename ScalesEstimatorType::Pointer scalesEstimator = ScalesEstimatorType::New();
        // scalesEstimator->SetMetric( similarity );
        // scalesEstimator->SetTransformForward( true );
        // scalesEstimator->SetSmallParameterVariation( 1.0 );

        // Optimizer config
        this->GetOptimizer()->SetNumberOfIterations(level_iterations[0]);
        // this->GetOptimizer()->SetScalesEstimator( scalesEstimator );
        if (verbose)
        {
            CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
            this->GetOptimizer()->AddObserver(itk::IterationEvent(), observer);
        };
    }

    void metric(std::string sim = "ssd", std::string sampling = "random", float percent = 0.2)
    {

        if (sim == "ssd")
            similarity = itk::MeanSquaresImageToImageMetricv4< TFixedImage, TMovingImage >::New();
        else if (sim == "cc")
            similarity = itk::CorrelationImageToImageMetricv4< TFixedImage, TMovingImage >::New();
        else if (sim == "mi")
            similarity = itk::MattesMutualInformationImageToImageMetricv4< TFixedImage, TMovingImage >::New();
        else if (sim == "cc_ants")
            similarity = itk::ANTSNeighborhoodCorrelationImageToImageMetricv4< TFixedImage, TMovingImage >::New();
        else
            printf("Undefined metric. Default ssd\n");

        this->SetMetric(similarity);

        // Sampling
        if (sampling == "random")
        {
            this->SetMetricSamplingStrategy(RegistrationType::MetricSamplingStrategyType::RANDOM);
            this->SetMetricSamplingPercentage(percent);
            if (verbose) printf("Sampling: Random\n");
        }
        else if (sampling == "regular")
        {
            this->SetMetricSamplingStrategy(RegistrationType::MetricSamplingStrategyType::REGULAR);
            this->SetMetricSamplingPercentage(percent);
            if (verbose) printf("Sampling: Regular\n");
        }
        else
        {
            this->SetMetricSamplingStrategy(RegistrationType::MetricSamplingStrategyType::NONE);
            if (verbose) printf("Sampling: None\n");
        };
    }

    template <unsigned int SplineOrder = 3>
    void bspline(unsigned int mesh_nodes = 8)
    {
        // Init transform
        // const unsigned int SplineOrder = 3;
        using TransformType     = itk::BSplineTransform< CoordinatesType, TFixedImage::ImageDimension, SplineOrder >;
        using InitializerType   = itk::BSplineTransformInitializer<TransformType, TFixedImage>;

        const TFixedImage * fixed = this->GetFixedImage();
        typename TransformType::Pointer transform = TransformType::New();

        typename InitializerType::Pointer init_trfm = InitializerType::New();
        typename TransformType::MeshSizeType mesh_size;
        mesh_size.Fill(mesh_nodes - SplineOrder);

        init_trfm->SetTransform(transform);
        init_trfm->SetImage(fixed);
        init_trfm->SetTransformDomainMeshSize(mesh_size);
        init_trfm->InitializeTransform();

        transform->SetIdentity();

        // Multiresolution levels
        unsigned int numberOfLevels = level_scales.size();
        typename RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
        typename RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
        shrinkFactorsPerLevel.SetSize(numberOfLevels);
        smoothingSigmasPerLevel.SetSize(numberOfLevels);
        for (unsigned int i = 0; i < numberOfLevels; ++i)
        {
            shrinkFactorsPerLevel[i] = level_scales[i];
            smoothingSigmasPerLevel[i] = level_sigmas[i];
        }
        
        // Multiresolution with bsplines
        typename RegistrationType::TransformParametersAdaptorsContainerType adaptors;

            // First, get fixed image physical dimensions
        typename TransformType::PhysicalDimensionsType fixedPhysicalDimensions;
        for (unsigned int i = 0; i < Self::ImageDimension; ++i)
        {
            fixedPhysicalDimensions[i] = fixed->GetSpacing()[i] *
                static_cast<double>(fixed->GetLargestPossibleRegion().GetSize()[i] - 1);
        }

            // Create the transform adaptors specific to B-splines
        for (unsigned int level = 0; level < numberOfLevels; ++level)
        {
            using ShrinkFilterType = itk::ShrinkImageFilter<TFixedImage, TFixedImage>;
            typename ShrinkFilterType::Pointer shrinkFilter = ShrinkFilterType::New();
            shrinkFilter->SetShrinkFactors(shrinkFactorsPerLevel[level]);
            shrinkFilter->SetInput(fixed);
            shrinkFilter->Update();

            // A good heuristic is to double the b-spline mesh resolution at each level
            typename TransformType::MeshSizeType requiredMeshSize;
            for (unsigned int d = 0; d < Self::ImageDimension; ++d)
            {
                requiredMeshSize[d] = mesh_size[d] << level;
            };

            using BSplineAdaptorType = itk::BSplineTransformParametersAdaptor<TransformType>;
            typename BSplineAdaptorType::Pointer bsplineAdaptor = BSplineAdaptorType::New();
            bsplineAdaptor->SetTransform(transform);
            bsplineAdaptor->SetRequiredTransformDomainMeshSize(requiredMeshSize);
            bsplineAdaptor->SetRequiredTransformDomainOrigin(
              shrinkFilter->GetOutput()->GetOrigin());
            bsplineAdaptor->SetRequiredTransformDomainDirection(
              shrinkFilter->GetOutput()->GetDirection());
            bsplineAdaptor->SetRequiredTransformDomainPhysicalDimensions(
              fixedPhysicalDimensions);
            adaptors.push_back(bsplineAdaptor);
        }


        // Config registration
        this->SetInitialTransform(transform);

            // Multiresolution
        this->SetNumberOfLevels(numberOfLevels);
        this->SetSmoothingSigmasPerLevel(smoothingSigmasPerLevel);
        this->SetShrinkFactorsPerLevel(shrinkFactorsPerLevel);
        this->SetTransformParametersAdaptorsPerLevel(adaptors);

            // Multiresolution event
        if (verbose)
        {
            using CommandType = RegistrationInterfaceCommand<Self>;
            typename CommandType::Pointer command = CommandType::New();
            this->AddObserver(itk::MultiResolutionIterationEvent(), command);
        }
        
    }
};

template <typename InputType, typename OutputType, typename TransformType>
typename OutputType::Pointer Resample(typename InputType::Pointer input, 
        typename OutputType::Pointer output, const TransformType * transform)
{
    using ResampleFilterType = itk::ResampleImageFilter< InputType, OutputType >;
    typename ResampleFilterType::Pointer resampler = ResampleFilterType::New();
    resampler->SetInput( input );
    resampler->SetTransform( transform );
    resampler->SetSize( output->GetLargestPossibleRegion().GetSize() );
    resampler->SetOutputOrigin(  output->GetOrigin() );
    resampler->SetOutputSpacing( output->GetSpacing() );
    resampler->SetOutputDirection( output->GetDirection() );
    resampler->SetDefaultPixelValue( 0 );
    resampler->Update();
    return resampler->GetOutput();
};

template <typename ImageType>
typename ImageType::Pointer ReadImage(const std::string & file)
{
    using ReaderType = itk::ImageFileReader<ImageType>;
    typename ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName(file);
    reader->Update();
    return reader->GetOutput();
};

template <typename ImageType>
void WriteImage(std::string file, typename ImageType::Pointer image)
{
    using WriterType = itk::ImageFileWriter<ImageType>;
    typename WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(file);
    writer->SetInput(image);
    writer->Update();
};

};