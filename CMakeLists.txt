cmake_minimum_required (VERSION 2.6)

project (mobilefacenet_caffe2ncnn)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -qopenmp")
#add executable file，添加要编译的可执行文件
add_executable(
    mobilefacenet_caffe2ncnn
    main.cpp
    mobilefacenet.cpp
    mobilefacenet.h
)

find_package(OpenCV REQUIRED)
target_link_libraries(mobilefacenet_caffe2ncnn ${OpenCV_LIBS})

set_target_properties(mobilefacenet_caffe2ncnn PROPERTIES CXX_STANDARD 11)
find_package(OpenMP)
#set environment variable，设置环境变量，编译用到的源文件全部都要放到这里，否则编译能够通过，但是执行的时候会出现各种问题
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

include_directories(./include/ncnn)
set_property(TARGET ncnn PROPERTY IMPORTED_LOCATION ../src/libncnn.a)
#add link library，添加可执行文件所需要的库
target_link_libraries(mobilefacenet_caffe2ncnn ncnn)
