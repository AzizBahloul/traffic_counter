cmake_minimum_required(VERSION 3.16)
project(traffic_counter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Allow user to override ONNXRUNTIME_ROOT if installed to custom path
if(NOT DEFINED ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_ROOT "/usr/local")
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

# OnnxRuntime include & lib (assumes installed under /usr/local or set ONNXRUNTIME_ROOT)
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

add_executable(traffic_counter
  src/main.cpp
  src/ObjectDetector.cpp
  src/TrafficCounter.cpp
  src/utils.cpp
)

target_include_directories(traffic_counter PRIVATE src ${ONNXRUNTIME_ROOT}/include)
target_link_libraries(traffic_counter PRIVATE ${OpenCV_LIBS} onnxruntime)
 # Link ONNX Runtime CUDA provider and define USE_CUDA if available
find_library(ORT_CUDA_LIB onnxruntime_providers_cuda PATHS ${ONNXRUNTIME_ROOT}/lib)
if(ORT_CUDA_LIB)
  message(STATUS "Found ONNX Runtime CUDA provider: ${ORT_CUDA_LIB}")
  target_link_libraries(traffic_counter PRIVATE ${ORT_CUDA_LIB})
  target_compile_definitions(traffic_counter PRIVATE USE_CUDA)
endif()

# Install target
install(TARGETS traffic_counter DESTINATION bin)
