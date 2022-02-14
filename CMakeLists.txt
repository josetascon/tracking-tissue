# Author: José D. Tascón-Vidarte <josedtascon@gmail.com>
# ===========================================
#               CMAKE Project
# ===========================================
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project( Tracking-Tissue )

# ===========================================
#               Find Libraries
# ===========================================

# Find ITK
find_package( ITK REQUIRED )
include(${ITK_USE_FILE})

# Find VTK
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

# Find OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Find Boost
find_package( Boost COMPONENTS program_options REQUIRED )
include_directories( ${Boost_INCLUDE_DIR} )

set(link_targets ${ITK_LIBRARIES} ${VTK_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})

# Optional OpenCL
option(ENABLE_OPENCL "Enable OpenCL" ON)
if(ENABLE_OPENCL)
    find_package( OpenCL REQUIRED )
    set(link_targets ${link_targets} OpenCL::OpenCL)
endif()

# Optional CUDA
option(ENABLE_CUDA "Enable CUDA" OFF)
if(ENABLE_CUDA)
    enable_language(CUDA)
    set (CMAKE_CUDA_ARCHITECTURES 86) #52 72 )
    set(link_targets ${link_targets} imart_cuda cufft)
endif()


# ===========================================
#               Command Build
# ===========================================

add_executable( tracking_imart_dem tracking_imart_dem.cpp )
target_link_libraries( tracking_imart_dem ${link_targets} )

add_executable( tracking_imart_tm tracking_imart_tm.cpp )
target_link_libraries( tracking_imart_tm ${link_targets} )

add_executable( tracking_imart_tmdem tracking_imart_tmdem.cpp )
target_link_libraries( tracking_imart_tmdem ${link_targets} )

add_executable( tracking_itk_bspline tracking_itk_bspline.cpp )
target_link_libraries( tracking_itk_bspline ${link_targets} )

add_executable( tracking_opencv tracking_opencv.cpp )
target_link_libraries( tracking_opencv ${link_targets} )