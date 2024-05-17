cmake_minimum_required(VERSION 3.0.0)
project( khlobustova )

set(OPENCV_DIR C:/Users/Svt/Desktop/khlobustova/opencv/build/install)

add_executable(lab1 prj.lab/lab01/lab01.cpp)
add_executable(lab2 prj.lab/lab02/lab02.cpp)
add_executable(lab3 prj.lab/lab03/lab03.cpp)
add_executable(lab4 prj.lab/lab04/lab04.cpp)
add_executable(lab5 prj.lab/lab05/lab05.cpp)
add_executable(lab6 prj.lab/lab06/lab06.cpp)
add_executable(lab7 prj.lab/lab07/lab07.cpp)
add_executable(lab8 prj.lab/lab08/lab08.cpp)

target_include_directories(lab1 PRIVATE ${OPENCV_DIR}/include)
target_link_libraries(lab1 ${OPENCV_DIR}/x64/vc17/lib/*.lib)

target_include_directories(lab2 PRIVATE ${OPENCV_DIR}/include)
target_link_libraries(lab2 ${OPENCV_DIR}/x64/vc17/lib/*.lib)

target_include_directories(lab3 PRIVATE ${OPENCV_DIR}/include)
target_link_libraries(lab3 ${OPENCV_DIR}/x64/vc17/lib/*.lib)

target_include_directories(lab4 PRIVATE ${OPENCV_DIR}/include)
target_link_libraries(lab4 ${OPENCV_DIR}/x64/vc17/lib/*.lib)

target_include_directories(lab5 PRIVATE ${OPENCV_DIR}/include)
target_link_libraries(lab5 ${OPENCV_DIR}/x64/vc17/lib/*.lib)

target_include_directories(lab6 PRIVATE ${OPENCV_DIR}/include)
target_link_libraries(lab6 ${OPENCV_DIR}/x64/vc17/lib/*.lib)

target_include_directories(lab7 PRIVATE ${OPENCV_DIR}/include)
target_link_libraries(lab7 ${OPENCV_DIR}/x64/vc17/lib/*.lib)

target_include_directories(lab8 PRIVATE ${OPENCV_DIR}/include)
target_link_libraries(lab8 ${OPENCV_DIR}/x64/vc17/lib/*.lib)
