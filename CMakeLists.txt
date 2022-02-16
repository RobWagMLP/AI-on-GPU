cmake_minimum_required(VERSION 3.16)

project(ClNN)


set(ClNNSrcs try.cpp)

add_executable(ClNN ${ClNNSrcs})

set(MathLibSrc "./cl_math_lib/cl_math_lib.cpp")
set(LayerSrc "./layer/enums.cpp" "./layer/basic_layer.cpp" "./layer/dense_layer.cpp" "./layer/dense_out.cpp")
set(ModelSrc "./model/stats.cpp" "./model/model.cpp")

add_library(mathLib  ${MathLibSrc})
add_library(layerLib ${LayerSrc}) 
add_library(modelLib ${ModelSrc})

find_package(OpenCL REQUIRED)

target_link_libraries(layerLib mathLib)
target_link_libraries(modelLib layerLib)

target_link_libraries(ClNN modelLib layerLib )

install(TARGETS ClNN DESTINATION bin)

add_dependencies(mathLib OpenCL::OpenCL)