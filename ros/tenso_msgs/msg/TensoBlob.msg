# TensoBlob — an opaque, self-describing Tenso wire-format packet.
#
# The payload is a complete Tenso packet (magic "TNSO", v4 header, body, and
# optional integrity footer). All shape/dtype/flags metadata lives *inside*
# the packet — decode it with tenso-core / include/tenso.h / cpp/tenso.hpp.
#
# Keeping the message to a single byte array (rather than mirroring shape/dtype
# into ROS fields) is what makes zero-copy intra-process transport possible via
# rclcpp::TypeAdapter<TensorView, TensoBlob>: the adapter just hands the bytes
# across without re-encoding. See tenso_ros for the adapter + A/B demo.

# Optional message timestamp / frame for sensor-style pipelines.
std_msgs/Header header

# The raw Tenso packet bytes.
uint8[] payload
