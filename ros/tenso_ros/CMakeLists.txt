cmake_minimum_required(VERSION 3.8)
project(tenso_ros)

# ---------------------------------------------------------------------------
# Linux-only (ROS2). Fail early on any other platform.
# ---------------------------------------------------------------------------
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR
    "tenso_ros is Linux-only (ROS2). Detected: ${CMAKE_SYSTEM_NAME}.")
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tenso_msgs REQUIRED)

# ---------------------------------------------------------------------------
# Tenso C ABI library + headers.
#
# The Tenso C++ shim (cpp/tenso.hpp) wraps the cbindgen-generated C ABI
# (include/tenso.h) and links against the tenso-ffi cdylib/staticlib produced
# by `cargo build -p tenso-ffi`. These live in the repo root relative to this
# package; override with -DTENSO_REPO_ROOT=/path/to/tenso if vendoring.
# ---------------------------------------------------------------------------
set(TENSO_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH
  "Path to the Tenso repository root (provides include/tenso.h, cpp/tenso.hpp).")

set(TENSO_INCLUDE_DIRS
  "${TENSO_REPO_ROOT}/include"   # include/tenso.h (C ABI)
  "${TENSO_REPO_ROOT}/cpp"       # cpp/tenso.hpp  (RAII shim)
)

# Locate the FFI shared/static library produced by cargo. Adjust paths to your
# cargo target dir / install layout as needed.
find_library(TENSO_FFI_LIB
  NAMES tenso_ffi libtenso_ffi
  HINTS
    "${TENSO_REPO_ROOT}/target/release"
    "${TENSO_REPO_ROOT}/target/debug"
    "$ENV{TENSO_FFI_LIB_DIR}"
)
if(NOT TENSO_FFI_LIB)
  message(WARNING
    "tenso-ffi library not found. Build it with `cargo build -p tenso-ffi` and "
    "set TENSO_FFI_LIB_DIR, or pass -DTENSO_FFI_LIB=/path/to/libtenso_ffi.so. "
    "Configuring anyway so the package layout is inspectable.")
endif()

# Helper to wire the Tenso includes + adapter + FFI lib onto a target.
function(tenso_ros_link_target tgt)
  target_include_directories(${tgt} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${TENSO_INCLUDE_DIRS}
  )
  ament_target_dependencies(${tgt}
    rclcpp rclcpp_components std_msgs sensor_msgs tenso_msgs)
  if(TENSO_FFI_LIB)
    target_link_libraries(${tgt} ${TENSO_FFI_LIB})
  endif()
endfunction()

# Publisher node.
add_executable(tensor_publisher src/tensor_publisher.cpp)
tenso_ros_link_target(tensor_publisher)

# Subscriber node.
add_executable(tensor_subscriber src/tensor_subscriber.cpp)
tenso_ros_link_target(tensor_subscriber)

install(TARGETS
  tensor_publisher
  tensor_subscriber
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
  DESTINATION include/)

ament_export_include_directories(include)

ament_package()
