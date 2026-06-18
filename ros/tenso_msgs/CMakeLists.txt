cmake_minimum_required(VERSION 3.8)
project(tenso_msgs)

# ---------------------------------------------------------------------------
# Tenso ROS packages are Linux-only (ROS2 is not supported on macOS/Windows in
# this project). Guard up front so accidental builds elsewhere fail loudly.
# ---------------------------------------------------------------------------
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR
    "tenso_msgs is Linux-only (ROS2). Detected: ${CMAKE_SYSTEM_NAME}.")
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TensoBlob.msg"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
