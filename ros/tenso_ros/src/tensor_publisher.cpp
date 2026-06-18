/*
 * tensor_publisher.cpp — SKELETON node that publishes Tenso tensors over the
 * TypeAdapter<TensorView, TensoBlob> path.
 *
 * It encodes a synthetic dense float32 tensor once per tick via the Tenso C
 * core (no Python, no manual byte layout) and publishes it as a
 * tenso_ros::TensorView. With intra-process comms enabled and a matching
 * adapted subscription, ROS moves the buffer through without re-serializing —
 * the "hello tensor" zero-copy demo. See README.md for the A/B comparison.
 *
 * Linux-only (ROS2). This is scaffolding: wire it into your launch/colcon
 * workspace; the encode/transport seams are the parts worth reviewing.
 */

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "rclcpp/rclcpp.hpp"

#include "tenso_ros/type_adapter.hpp"

using namespace std::chrono_literals;

namespace tenso_ros {

class TensorPublisher : public rclcpp::Node
{
public:
  explicit TensorPublisher(const rclcpp::NodeOptions & options)
  : rclcpp::Node("tenso_tensor_publisher", options)
  {
    // Declare the publisher in terms of the *adapted* type. The transported
    // wire message remains tenso_msgs::msg::TensoBlob.
    pub_ = this->create_publisher<TensoTensorAdapted>("tenso/tensor", 10);

    timer_ = this->create_wall_timer(
      100ms, [this]() { this->tick(); });

    RCLCPP_INFO(this->get_logger(), "tenso tensor publisher up on 'tenso/tensor'");
  }

private:
  void tick()
  {
    // Synthetic [1, 8] float32 tensor; replace with real sensor/model output.
    const std::vector<std::uint32_t> shape{1, 8};
    std::vector<float> data(8);
    for (std::size_t i = 0; i < data.size(); ++i) {
      data[i] = static_cast<float>(seq_) + static_cast<float>(i) * 0.5f;
    }

    // Encode once via the Tenso C core (Mode A, caller-allocates).
    auto tensor = TensorView::encode_dense(
      data.data(),
      data.size() * sizeof(float),
      shape,
      tenso::DType::Float32);

    RCLCPP_DEBUG(
      this->get_logger(), "publishing seq=%lu (%zu packet bytes)",
      static_cast<unsigned long>(seq_), tensor.packet().size());

    // Move the adapted value in; with intra-process comms this hands the
    // buffer to the subscriber without a re-encode.
    pub_->publish(std::move(tensor));
    ++seq_;
  }

  rclcpp::Publisher<TensoTensorAdapted>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::uint64_t seq_ = 0;
};

}  // namespace tenso_ros

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  // use_intra_process_comms(true) is what unlocks the zero-copy path when the
  // subscriber lives in the same process/container.
  rclcpp::NodeOptions options;
  options.use_intra_process_comms(true);
  rclcpp::spin(std::make_shared<tenso_ros::TensorPublisher>(options));
  rclcpp::shutdown();
  return 0;
}
