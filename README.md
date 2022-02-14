# Tracking Algorithms for Tumors and Organs

This is a collection of real-time algorithms for tracking tumors and organs on 2D Cine-MRIs.

## Description:

We implemented nine (9) tracking algorithms based on deformable image registration (DIR), object tracking, and image segmentation. The DIR algorithms find a transformation between the reference and each input image from the Cine-MRI. The resulting transform is then applied to the reference tumor mask to obtain the predicted mask. Computer vision proposals solved object tracking with a group of algorithms called tracking by detection. Here a classifier model locates the object, and then a model update is produced for each frame. A bounding box represents the object's location within the image. In our case, we desire the tumor mask, and therefore we transform the reference tumor mask to fit the output bounding box. The segmentation algorithms work as individual trackers running in parallel for the organ and the tumor. The segmentation algorithms are pre-trained first and afterward only use the input image intensities to predict the masks. 

The algorithms are: Demons registration (DEM), B-spline registration (BSP), Template Matching (TM), Template Matching with Demons Registration (TMDEM), U-net segmentation (UNET), Multiple Instance Learning (MIL), Kernelized Correlation Filter (KCF), Discriminative Correlation Filter with Channel and Spatial Reliability (CSRT), Siamese region proposal network for tracking (DRPN). 

## Dependencies:
* ITK (read/write images)
* VTK (visualize images)
* Boost (program options)
* OpenCV (some tracking algorithms)
* [imart](https://github.com/josetascon/imart)

## Build

In the project folder use cmake to build.

mkdir build\
cd build\
cmake ..

## Run

The scripts "./tracking_imart_dem, ./tracking_imart_tm, ./tracking_imart_tmdem" contain the DEM, TM, and TMDEM algorithms respectively.

The script "./tracking_opencv" collects the tracking algorithms MIL, KCF, CSRT, DRPN.

The UNET algorithm is located [here](https://github.com/josetascon/unet).

Run a tracking script as: 
>./tracking_imart_dem -i <input_folder> -o <output_folder> -vp

The input folder requires a folder with the base segmentation and the raw images.


Citation
--------

To cite when using this toolbox, please reference, as appropriate:\
@article{tascon2022real,\
  title={Real-time algorithms for tracking tumor and organs in 2D Cine-MRI},\
  author={},\
  booktitle={},\
  pages={},\
  year={2022},\
  organization={}\
}

