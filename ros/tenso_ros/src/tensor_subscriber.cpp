/*
 * tensor_subscriber.cpp — SKELETON node that subscribes to Tenso tensors via
 * the TypeAdapter<TensorView, TensoBlob> path and decodes them zero-copy.
 *
 * The callback receives a tenso_ros::TensorView (the adapter converted the
 * TensoBlob for us). We decode it with the Tenso C++ shim (tenso::read), which
 * yields a tenso::Packet whose body *borrows* from the received buffer — no
 * tensor-data copy. See README.md for the A/B demo vs sensor_msgs/Image.
 *
 * Linux-only (ROS2). Scaffolding: the decode + lifetime seam (the borrowed
 * body must not outlive the received buffer) is the part to review.
 */

#include <cstdint>
#include <memory>

#include "rclcpp/rclcpp.hpp"

#include "tenso_ros/type_adapter.hpp"

namespace tenso_ros {

class TensorSubscriber : public rclcpp::Node
{
public:
  explicit TensorSubscriber(const rclcpp::NodeOptions & options)
  : rclcpp::Node("tenso_tensor_subscriber", options)
  {
    sub_ = this->create_subscription<TensoTensorAdapted>(
      "tenso/tensor", 10,
      [this](const tenso_ros::TensorView & tensor) { this->on_tensor(tensor); });

    RCLCPP_INFO(this->get_logger(), "tenso tensor subscriber up on 'tenso/tensor'");
  }

private:
  void on_tensor(const tenso_ros::TensorView & tensor)
  {
    try {
      // Zero-copy decode: pkt.body_ptr() points into tensor.packet().
      tenso::Packet pkt = tensor.view();
      const auto shape = pkt.shape();
      const float * vals = pkt.data_as<float>();

      RCLCPP_INFO(
        this->get_logger(),
        "got tensor dtype=%s ndim=%zu nelem=%zu first=%f",
        tenso::dtype_name(pkt.dtype()),
        pkt.ndim(),
        pkt.num_elements(),
        (pkt.num_elements() > 0 && vals != nullptr) ? static_cast<double>(vals[0]) : 0.0);

      (void)shape;  // shape is available for downstream consumers.
    } catch (const tenso::Error & e) {
      RCLCPP_ERROR(this->get_logger(), "tenso decode failed: %s", e.what());
    }
  }

  rclcpp::Subscription<TensoTensorAdapted>::SharedPtr sub_;
};

}  // namespace tenso_ros

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.use_intra_process_comms(true);
  rclcpp::spin(std::make_shared<tenso_ros::TensorSubscriber>(options));
  rclcpp::shutdown();
  return 0;
}
