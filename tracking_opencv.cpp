/*
* @Author: jose
* @Date:   2021-07-14 13:11:41
* @Last Modified by:   jose
* @Last Modified time: 2021-10-06 15:55:08
*/

// std libs
#include <iostream>
#include <sstream>
#include <filesystem>
#include <thread>

// boost libs
#include <boost/program_options.hpp>

// opencv libs
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// local libs
#include "itk_opencv.hpp"

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
    std::string path_model = "./";
    std::string file_mask;
    std::string tracking_method;
    bool verbose, plot, write;
    double fps, tf;
    int extra_pixels;
    std::vector<int> write_vector;

    // Program options
    po::options_description desc("Tracking 2D CineMR images with opencv methods. Options");
    desc.add_options()
    ("help,h", "Help message")
    ("input,i", po::value<std::string>(&path_input), "Input folder")
    ("mask,m", po::value<std::string>(&file_mask), "Mask image used for tracking")
    ("output,o", po::value<std::string>(&path_output), "Output folder")
    ("method,t", po::value<std::string>(&tracking_method)->default_value("KCF"), 
        "Tracking Method. Options: CSRT, DaSiamRPN, GOTURN, KCF, MIL")
    ("models,z", po::value<std::string>(&path_model)->multitoken(), "Path of models")
    ("write-vector,y", po::value<std::vector<int>>(&write_vector)->multitoken(), "Write vector")
    ("verbose,v", po::bool_switch(&verbose), "Enable verbose")
    ("plot,p", po::bool_switch(&plot), "Enable plot")
    ("frame_per_second,f", po::value<double>(&fps)->default_value(4.0), "Images frames per second")
    ("extra-pixels,x", po::value<int>(&extra_pixels)->default_value(4), "Extra pixels of bounding box");

    // Parse command line options
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")){ std::cerr << desc << std::endl; return 1; };

    if (vm.count("input")) std::cout << "Folder input: " << path_input << std::endl << std::endl;
    else {std::cerr << "Please provide input folder with images" << std::endl; return 1; };

    if (vm.count("mask")) std::cout << "File mask: " << path_input << std::endl << std::endl;
    else {std::cerr << "Please provide image mask or template" << std::endl; return 1; };

    std::cout << "Tracking method: " << tracking_method << std::endl;

    if (path_output == "") write = false;
    else write = true;

    tf = 1000/fps; // sampling time in milliseconds

    // ============================================
    //       Testing video tracking with opencv
    // ============================================

    cv::Ptr<cv::Tracker> tracker;

    #if (CV_MINOR_VERSION < 3)
    {
        tracker = cv::Tracker::create(tracking_method);
    }
    #else
    {
        if (tracking_method == "CSRT")
            tracker = cv::TrackerCSRT::create();
        else if (tracking_method == "DaSiamRPN")
        {
            cv::TrackerDaSiamRPN::Params params;
            params.model       = path_model + "/" + "dasiamrpn_model.onnx";
            params.kernel_r1   = path_model + "/" + "dasiamrpn_kernel_r1.onnx";
            params.kernel_cls1 = path_model + "/" + "dasiamrpn_kernel_cls1.onnx";
            tracker = cv::TrackerDaSiamRPN::create(params);
        }
        else if (tracking_method == "GOTURN")
            tracker = cv::TrackerGOTURN::create();
        else if (tracking_method == "KCF")
            tracker = cv::TrackerKCF::create();
        else if (tracking_method == "MIL")
            tracker = cv::TrackerMIL::create();
        else
        {
            std::cout << "Unknown tracking method" << std::endl; 
            return 1;
        }

        // if (trackerType == "BOOSTING")
        //     tracker = cv::TrackerBoosting::create();
        // if (trackerType == "TLD")
        //     tracker = cv::TrackerTLD::create();
        // if (trackerType == "MEDIANFLOW")
        //     tracker = cv::TrackerMedianFlow::create();
        // if (trackerType == "MOSSE")
        //     tracker = cv::TrackerMOSSE::create();
    }
    #endif

    // Images
    auto img_fixed = image_cpu<type>::new_pointer();
    auto img_input = image_cpu<type>::new_pointer();
    auto img_mask = image_cpu<type>::new_pointer();
    auto img_mask_warped = image_cpu<type>::new_pointer();
    
    // Read fixed image
    auto list_files_images = list_directory(path_input);
    std::string file_fixed = list_files_images.front();
    img_fixed->read(file_fixed);
    cv::Mat img_frame = read_with_itk_to_opencv<unsigned char>(file_fixed);
    cv::Mat img_color;
    std::cout << "Read fixed: " << file_fixed << std::endl;

    img_mask->read(file_mask);
    std::cout << "Read tracking mask: " << file_mask << std::endl << std::endl;
    
    // Bounding Box
    auto bbox_fixed = bounding_box(img_mask, extra_pixels);
    auto img_mask_region = img_mask->region(bbox_fixed[0], bbox_fixed[1]);
    auto mask_resolution = resolution<type,vector_cpu<type>>::new_pointer(img_mask_region);
    printf("box %d, %d, %d, %d\n", bbox_fixed[0][0], bbox_fixed[0][1], bbox_fixed[1][0], bbox_fixed[1][1] );

    // Plot
    auto view = viewer_track<image_type>::new_pointer();
    auto img_view_input = img_fixed->copy();
    auto img_view_mask = img_mask->copy();
    
    // Setup viewer
    if (plot)
    {
        view->add_image(img_view_input);
        view->add_image(img_view_mask);
        view->setup();
    }

    timer t("ms");
    t.start();
    double now_time = 0.0;
    double sum_time = 0.0;

    // Output folder
    std::string base = std::string(fs::path(file_mask).parent_path().filename());
    std::string folder_output = path_output + "/" + base;
    if (write)
    {
        std::cout << "Create output folder: " << folder_output << std::endl;
        std::cout << std::endl;
        fs::create_directories(folder_output);
    }

    // Bounding Box
    cv::Rect cv_bbox_fixed = cv::Rect(bbox_fixed[0][0], bbox_fixed[0][1], 
        bbox_fixed[1][0], bbox_fixed[1][1]);
    
    cv::cvtColor(img_frame, img_color, cv::COLOR_GRAY2BGR);
    // tracker->init(img_frame, cv_bbox_fixed);
    tracker->init(img_color, cv_bbox_fixed);

    std::vector<std::vector<int>> bbox_moving = bbox_fixed;

    int ww = 0;

    // Main Loop
    for(size_t i = 1; i < list_files_images.size(); i++ )
    {
        // Read new input
        std::string file_input = list_files_images[i];
        img_input->read(file_input);
        if (verbose) std::cout << "Read input: " << file_input << std::endl;
        std::string num = fs::path(file_input).stem();
        num = num.substr(num.size()-4, num.size());
        std::string ext = fs::path(file_input).extension();
        // std::cout << "Number: " << num << std::endl;

        img_frame = read_with_itk_to_opencv<unsigned char>(file_input);
        cv::cvtColor(img_frame, img_color, cv::COLOR_GRAY2BGR);

        if (verbose)
        {
            t.lap(); 
            now_time = t.get_elapsed();
            sum_time += now_time;
            printf("Read time: \t%5.2f [ms]\n", now_time);
        }
        
        // Update the tracking result
        // bool ok = tracker->update(img_frame, cv_bbox_fixed);
        bool ok = tracker->update(img_color, cv_bbox_fixed);

        // bool ok = false;
        // try
        // {
        //     ok = tracker->update(img_frame, cv_bbox_fixed);
        // }
        // catch (...)
        // {
        //     printf("Fail\n");
        // }
        
        if (verbose)
        {
            t.lap(); 
            now_time = t.get_elapsed();
            sum_time += now_time;
            printf("Algorithm time: %5.2f [ms]\n", now_time);
        }

        // Warp output images
        if (ok)
        {
            bbox_moving[0][0] = cv_bbox_fixed.x;
            bbox_moving[0][1] = cv_bbox_fixed.y;
            bbox_moving[1][0] = cv_bbox_fixed.width;
            bbox_moving[1][1] = cv_bbox_fixed.height;
            printf("box %d, %d, %d, %d\n", bbox_moving[0][0], bbox_moving[0][1], bbox_moving[1][0], bbox_moving[1][1] );

            if (cv_bbox_fixed.x < 0)
            {
                bbox_moving[0][0] = 0;
                bbox_moving[1][0] += cv_bbox_fixed.x;
            }

            if (cv_bbox_fixed.y < 0)
            {
                bbox_moving[0][1] = 0;
                bbox_moving[1][1] += cv_bbox_fixed.y;
            }
        };

        if ((cv_bbox_fixed.width < 0.2*bbox_fixed[1][0]) or (cv_bbox_fixed.height < 0.2*bbox_fixed[1][1]))
        {
            printf("defualt bbox\n");
            bbox_moving = bbox_fixed;
        }

        auto tmp_img_mask_region = img_mask_region;
        if ((bbox_moving[1][0] !=  bbox_fixed[1][0]) or (bbox_moving[1][1] != bbox_fixed[1][1]))
        {
            printf("diff size\n");
            tmp_img_mask_region = mask_resolution->apply(bbox_moving[1]);

        }
        std::vector<int> pre = bbox_moving[0];
        std::vector<int> post(bbox_moving[1].size(),0);
        for (int k = 0; k < img_fixed->get_dimension(); k++)
        {
            post[k] = img_fixed->get_size()[k] - bbox_moving[0][k] - bbox_moving[1][k];
        }

        // Change img_mask_region in case of scaling in tracking
        // auto img_mask_warped = pad(img_mask_region, pre, post);
        auto img_mask_warped = pad(tmp_img_mask_region, pre, post);
        img_mask_warped->set_spacing(img_mask->get_spacing());
        img_mask_warped->set_origin(img_mask->get_origin());
        img_mask_warped->set_direction(img_mask->get_direction());
        // auto img_mask_region = img_mask_warped->region(bbox_moving[0], bbox_moving[1]);
        
        std::string file_output = folder_output + "/" + base + "_" + num + ext;
        if (write)
        {
            bool valid = true;
            if ((write_vector.size() > 0) or (write_vector.size() < ww))
            {
                valid = false;
                if (i == write_vector[ww])
                {
                    ww += 1;
                    valid = true;
                }
            }
            if (valid)
            {
                std::cout << "Write output: " << file_output << std::endl;
                img_mask_warped->write(file_output);
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
            img_view_mask->equal(*img_mask_warped);
            view->update(0);
            view->update(1);
            view->render();

            if (verbose)
            {
                t.lap();
                now_time = t.get_elapsed();
                sum_time += now_time;
                printf("Viewer time: \t%5.2f [ms]\n", now_time);
            }
        }

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
        
    }
    return 0;
}

std::vector<std::string> list_directory(std::string path)
{
    std::list<std::string> list_files;
    for (const auto & entry : fs::directory_iterator(path))
        list_files.push_back(entry.path());
    list_files.sort();
    return std::vector<std::string>(list_files.begin(), list_files.end());
};