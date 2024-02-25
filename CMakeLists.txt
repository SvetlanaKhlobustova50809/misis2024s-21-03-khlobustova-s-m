cmake_minimum_required(VERSION 3.0.0)
project( khlobustova )

set(OPENCV_DIR C:/Users/Svt/Desktop/khlobustova/opencv/build/install)

add_executable(${PROJECT_NAME} prj.lab/lab1.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${OPENCV_DIR}/include)

target_link_libraries(${PROJECT_NAME} ${OPENCV_DIR}/x64/vc17/lib/opencv_core490d.lib ${OPENCV_DIR}/x64/vc17/lib/opencv_imgcodecs490d.lib ${OPENCV_DIR}/x64/vc17/lib/opencv_highgui490d.lib)
