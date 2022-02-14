/*
* @Author: jose
* @Date:   2021-09-14 19:41:47
* @Last Modified by:   jose
* @Last Modified time: 2022-02-13 21:35:05
*/

// std libs
#include <iostream>
#include <memory>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <thread>

// boost libs
#include <boost/program_options.hpp>

// local libs
#include "itk_registration.hpp"

// imart libs
#include "imart/image.h"
#include "imart/image_utils.h"
#include "imart/viewer_track.h"
#include "imart/resolution.h"
#include "imart/utils/timer.h"

using namespace imart;
namespace po = boost::program_options;
namespace fs = std::filesystem;

std::vector<std::string> list_directory(std::string path);

int main(int argc, char **argv)
{
    using type = float;
    // using type = double;
    using image_type = image_cpu<type>;
    
    // Variables
    std::string path_input;
    std::string path_output = "";
    std::string similarity;
    std::string sampling;
    bool verbose, plot, write, cine, continuous;
    unsigned int mesh_size;
    float sampling_percent;
    double fps, tf;
    
    std::vector<int> level_scales({8,4});
    std::vector<int> level_sigmas({3,1});
    std::vector<int> level_iterations({60,40});
    std::vector<int> write_vector;

    // Program options
    po::options_description desc("Tracking 2D CineMR images with opencv methods. Options");
    desc.add_options()
    ("help,h", "Help message")
    ("input,i", po::value<std::string>(&path_input), "Input folder")
    ("output,o", po::value<std::string>(&path_output), "Output folder")
    ("cine", po::bool_switch(&cine), "Main folder with input images sequence is cine/ instead of image/")
    ("continuous,c", po::bool_switch(&continuous), "Enable tracking with continuous mode")
    ("write-vector,y", po::value<std::vector<int>>(&write_vector)->multitoken(), "Write vector")
    ("frame-per-second,f", po::value<double>(&fps)->default_value(4.0), "Images frames per second")
    ("metric,t", po::value<std::string>(&similarity)->default_value("ssd"), 
        "Similarity Metric. Options: ssd, cc, mi, cc_ants")
    ("bspline,b", po::value<unsigned int>(&mesh_size)->default_value(8), "Bspline mesh size")
    ("sampling,z", po::value<std::string>(&sampling)->default_value("random"), 
        "Metric sampling strategy. Options: none, regular, random")
    ("sampling-percent,n", po::value<float>(&sampling_percent)->default_value(0.5), "Sampling percentage")
    ("opt-level-scales,s", po::value<std::vector<int>>(&level_scales)->multitoken(), "Optimizer multilevel scales")
    ("opt-level-sigmas,g", po::value<std::vector<int>>(&level_sigmas)->multitoken(), "Optimizer multilevel smoothing sigmas")
    ("opt-level-iterations,k", po::value<std::vector<int>>(&level_iterations)->multitoken(), "Optimizer multilevel iterations")
    ("verbose,v", po::bool_switch(&verbose), "Enable verbose")
    ("plot,p", po::bool_switch(&plot), "Enable plot");

    // Parse command line options
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")){ std::cerr << desc << std::endl; return 1; };

    if (level_scales.size() != level_iterations.size()) {std::cerr << "Multilevel scales size is different to multilevel iterations" << std::endl; return 1; };

    if (vm.count("input")) std::cout << "Folder input: " << path_input << std::endl << std::endl;
    else {std::cerr << "Please provide input folder with images" << std::endl; return 1; };

    std::cout << "Similarity metric: " << similarity << std::endl;

    if (path_output == "") write = false;
    else write = true;

    tf = 1000/fps; // sampling time in milliseconds

    // ============================================
    //          Testing video registration
    // ============================================

    std::string path_image;
    std::vector<std::string> path_organs;

    // Read folders in directory
    auto list_folders = list_directory(path_input);

    for (auto const& folder : list_folders)
    {
        std::string relative = fs::path(folder).filename();
        if (relative == "boundary")
            continue;
        else if (relative.find("image") != std::string::npos)
        {
            if (not cine) { path_image = folder; };
        }
        else if (relative.find("cine") != std::string::npos)
        {
            if (cine) { path_image = folder; };
        }
        else if ( not fs::is_directory(folder) )
            ;
        else
            path_organs.push_back(folder);
    }

    const unsigned int Dimension = 2;
    constexpr unsigned int SplineOrder = 3;
    using CoordinatesType = double;
    using ImageType = itk::Image<type, Dimension>;
    using ReaderType = itk::ImageFileReader<ImageType>;
    using TransformType = itk::Transform<CoordinatesType,Dimension,Dimension>;
    using RegistrationType = itk::RegistrationBsplines< ImageType, ImageType >;
    
    // Images
    auto img_fixed = image_cpu<type>::new_pointer();
    auto img_input = image_cpu<type>::new_pointer();
    std::vector<image_cpu<type>::pointer> img_organs(path_organs.size());
    std::vector<image_cpu<type>::pointer> img_organs_warped(path_organs.size());
    std::vector<ImageType::Pointer> itk_organs(path_organs.size());
    std::vector<ImageType::Pointer> itk_organs_warped(path_organs.size());

    for (int k = 0; k < path_organs.size(); k++)
    {
        img_organs[k] = image_cpu<type>::new_pointer();
        img_organs_warped[k] = image_cpu<type>::new_pointer();
        itk_organs[k] = ImageType::New();
        itk_organs_warped[k] = ImageType::New();
    }
    
    // Read fixed image
    auto list_files_images = list_directory(path_image);
    std::string file_fixed = list_files_images.front();
    img_fixed->read(file_fixed);
    ImageType::Pointer itk_fixed = itk::ReadImage<ImageType>(file_fixed);
    std::cout << "Read fixed: " << file_fixed << std::endl;

    for(int k = 0; k < path_organs.size(); k++)
    {
        auto list_files_organ = list_directory(path_organs[k]);
        img_organs[k]->read(list_files_organ.front());
        itk_organs[k] = itk::ReadImage<ImageType>(list_files_organ.front());
        std::cout << "Read " << std::string(fs::path(path_organs[k]).filename()) 
            << ": " << list_files_organ.front() << std::endl;
    }
    std::cout << std::endl;

    // Registration
    RegistrationType::Pointer registration = RegistrationType::New();
    registration->SetFixedImage(itk_fixed);
    registration->SetMovingImage(itk_fixed); // tmp setup
    // registration->set_verbose(verbose);
    registration->set_verbose(false);
    registration->set_levels(level_iterations, level_scales, level_sigmas);
    registration->metric(similarity);
    registration->optimizer();

    // Plot
    auto view = viewer_track<image_type>::new_pointer();
    auto img_view_input = img_fixed->clone();
    std::vector<image_cpu<type>::pointer> img_view_organs(path_organs.size());
    for(int k = 0; k < path_organs.size(); k++)
        img_view_organs[k] = img_organs[k]->clone();
    
    // Setup viewer
    if (plot)
    {
        view->add_image(img_view_input);
        for(int k = 0; k < path_organs.size(); k++) 
            view->add_image(img_view_organs[k]);
        view->setup();
    }

    timer t("ms");
    t.start();
    double now_time = 0.0;
    double sum_time = 0.0;

    // Output folder
    std::vector<std::string> folders_output(path_organs.size());
    if (write)
    {
        for(int k = 0; k < path_organs.size(); k++) 
        {
            std::string folder_output = path_output + "/" + std::string(fs::path(path_organs[k]).filename());
            std::cout << "Create output folder: " << folder_output << std::endl;
            fs::create_directories(folder_output);
            folders_output[k] = folder_output;
        }
        std::cout << std::endl;
    }

    int ww = 0;

    // continuous variables
    bool use_fixed = true;
    auto img_previous = image_cpu<type>::new_pointer();
    auto itk_previous = itk_fixed;

    // Main Loop
    for(size_t i = 1; i < list_files_images.size(); i++ )
    {
        // Read new input
        std::string file_input = list_files_images[i];
        img_input->read(file_input);
        if (verbose) std::cout << "Read input: " << file_input << std::endl;
        std::string num = fs::path(file_input).stem();
        num = num.substr(num.size()-4,num.size());
        std::string ext = fs::path(file_input).extension();
        // std::cout << "Number: " << num << std::endl;

        ImageType::Pointer itk_input = ImageType::New();
        img_input->write_itk<Dimension>(itk_input);

        if (verbose)
        {
            t.lap(); 
            now_time = t.get_elapsed();
            sum_time += now_time;
            printf("Read time: \t%5.2f [ms]\n", now_time);
        }

        // Registration
        registration->SetFixedImage(itk_input);
        if (use_fixed) registration->SetMovingImage(itk_fixed);
        else registration->SetMovingImage(itk_previous);
        registration->metric(similarity);
        registration->bspline<SplineOrder>(mesh_size); // init transform
        registration->Update();

        if (verbose)
        {
            t.lap(); 
            now_time = t.get_elapsed();
            sum_time += now_time;
            printf("Register time: \t%5.2f [ms]\n", now_time);
        }

        // Warp output images
        auto transformation = registration->GetOutput()->Get();
        
        for(size_t k = 0; k < img_organs.size(); k++)
        {
            // auto img_mask_warped = warp *****
            if (use_fixed)
                itk_organs_warped[k] = itk::Resample<ImageType, ImageType, TransformType>(itk_organs[k], itk_input, transformation);
            else
                itk_organs_warped[k] = itk::Resample<ImageType, ImageType, TransformType>(itk_organs_warped[k], itk_input, transformation);
            img_organs_warped[k]->read_itk<Dimension>( itk_organs_warped[k] );
            
            
            // Write output file
            std::string base = fs::path(path_organs[k]).filename();
            std::string file_output = folders_output[k] + "/" + base + "_" + num + ext;
            if (write)
            {
                bool valid = true;
                if ((write_vector.size() > 0) or (write_vector.size() < ww))
                {
                    valid = false;
                    if (i == write_vector[ww])
                    {
                        if (k == itk_organs.size() - 1) { ww += 1; };
                        valid = true;
                    }
                }
                if (valid)
                {
                    std::cout << "Write output: " << file_output << std::endl;
                    itk::WriteImage<ImageType>(file_output, itk_organs_warped[k]);
                }
            }

        }

        if (verbose)
        {
            t.lap(); 
            now_time = t.get_elapsed();
            sum_time += now_time;
            printf("Warp time: \t%5.2f [ms]\n", now_time);
        }

        // Plot
        if (plot)
        {
            img_view_input->equal(*img_input);
            view->update(0);
            
            for(int k = 0; k < path_organs.size(); k++) 
            {
                img_view_organs[k]->equal(*img_organs_warped[k]);
                view->update(k+1);
            }
            
            view->render();

            if (verbose)
            {
                t.lap();
                now_time = t.get_elapsed();
                sum_time += now_time;
                printf("Viewer time: \t%5.2f [ms]\n", now_time);
            }
        }

        // Update when tracking continuous image sequences
        if (continuous)
        {
            use_fixed = false;
            img_previous = img_input->clone();
            itk_previous = itk_input;
            
            type ccfm = cross_correlation(img_fixed, img_input);
            // printf("cc %f\n",ccfm);
            if (ccfm > 0.97){ use_fixed = true; };
        };

        if (verbose)
        {
            t.lap(); 
            now_time = t.get_elapsed();
            sum_time += now_time;
            printf("Total time: \t%5.2f [ms]\n", sum_time);
        }

        if (sum_time < tf)
            std::this_thread::sleep_for(std::chrono::milliseconds( int(tf-sum_time) ));

        // update for next iteration
        t.lap();
        sum_time = 0.0;
    };

    return 0;
};

std::vector<std::string> list_directory(std::string path)
{
    std::list<std::string> list_files;
    for (const auto & entry : fs::directory_iterator(path))
        list_files.push_back(entry.path());
    list_files.sort();
    return std::vector<std::string>(list_files.begin(), list_files.end());
};

