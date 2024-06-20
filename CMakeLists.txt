cmake_minimum_required(VERSION 3.0.0)
project(khlobustova)

set(OPENCV_DIR C:/Users/Svt/Desktop/khlobustova/opencv/build)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "cmake")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/bin.dbg)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/bin.rel)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_SOURCE_DIR}/bin.relwithdbg)

find_package(OpenCV CONFIG REQUIRED)

add_subdirectory(prj.lab)
add_subdirectory(prj.cw)

#install(TARGETS ${PROJECT_NAME} DESTINATION ./exefile)

install(FILES prj.cw/input/input_layout1.txt
prj.cw/input/input_layout2.txt
prj.cw/input/test1.png
prj.cw/input/test2.png
prj.cw/input/test3.png
prj.cw/input/test4.png
prj.cw/input/layout_data1.txt
prj.cw/input/layout_data2.txt
DESTINATION ./prj.cw/input)

install(FILES prj.cw/output/src_data1.txt
prj.cw/output/src_data2.txt
prj.cw/output/src1.txt
prj.cw/output/src2.txt
DESTINATION ./prj.cw/output)

set_target_properties(${EDITOR_PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")


