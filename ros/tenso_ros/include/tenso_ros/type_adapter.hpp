/*
 * type_adapter.hpp — rclcpp::TypeAdapter mapping a custom C++ tensor type to
 * the tenso_msgs::msg::TensoBlob wire message.
 *
 * The whole point of the adapter is zero-copy intra-process transport: a
 * publisher hands the node a `tenso_ros::TensorView` (a thin owner of an
 * encoded Tenso packet); rclcpp moves the bytes into a TensoBlob and, when the
 * subscription is intra-process and uses the *same* adapted type, delivers the
 * same buffer back without re-encoding or re-serializing.
 *
 * Encoding/decoding is delegated to the Tenso C/C++ shim (cpp/tenso.hpp over
 * include/tenso.h). This file never touches the wire format byte layout itself.
 *
 * Linux-only (ROS2). See README.md for the A/B demo vs sensor_msgs/Image.
 */

#ifndef TENSO_ROS__TYPE_ADAPTER_HPP_
#define TENSO_ROS__TYPE_ADAPTER_HPP_

#include <cstdint>
#include <utility>
#include <vector>

#include "rclcpp/type_adapter.hpp"
#include "tenso_msgs/msg/tenso_blob.hpp"

// The Tenso C++ shim (RAII over the C ABI). Provides tenso::Packet,
// tenso::read(), tenso::write(), tenso::DType.
#include "tenso.hpp"

namespace tenso_ros {

// ---------------------------------------------------------------------------
// TensorView — the custom ("application") type the adapter maps to/from.
//
// It owns an *already-encoded* Tenso packet (a std::vector<uint8_t>). Producing
// the encoded bytes once (via tenso::write) and then moving them through ROS is
// what keeps the hot path copy-free. Use tenso::read(packet()) on the consumer
// side to get a zero-copy tenso::Packet over the body.
//
// (Named "TensorView" to match the contract's TypeAdapter<TensorView, TensoBlob>
// wording; it is an owning value type, not a borrowing view.)
// ---------------------------------------------------------------------------
class TensorView {
public:
  TensorView() = default;
  explicit TensorView(std::vector<uint8_t> packet)
  : packet_(std::move(packet)) {}

  // Build directly from a dense source buffer (encodes via the C core).
  static TensorView encode_dense(
    const void * data,
    std::size_t data_bytes,
    const std::vector<std::uint32_t> & shape,
    tenso::DType dtype,
    std::size_t alignment = tenso::DEFAULT_ALIGNMENT,
    bool check_integrity = false,
    bool compress = false)
  {
    return TensorView(
      tenso::write(
        data, data_bytes, shape, dtype, alignment, check_integrity, compress));
  }

  // Decode into a zero-copy tenso::Packet (body borrows from this->packet_,
  // which must outlive the returned Packet).
  tenso::Packet view() const { return tenso::read(packet_.data(), packet_.size()); }

  const std::vector<uint8_t> & packet() const { return packet_; }
  std::vector<uint8_t> & packet() { return packet_; }

private:
  std::vector<uint8_t> packet_;
};

}  // namespace tenso_ros

// ---------------------------------------------------------------------------
// The TypeAdapter specialization. With this in scope, a publisher/subscription
// declared as rclcpp::Publisher<TensoTensorAdapted> accepts/yields
// tenso_ros::TensorView while transporting tenso_msgs::msg::TensoBlob.
// ---------------------------------------------------------------------------
template<>
struct rclcpp::TypeAdapter<tenso_ros::TensorView, tenso_msgs::msg::TensoBlob>
{
  using is_specialized = std::true_type;
  using custom_type = tenso_ros::TensorView;
  using ros_message_type = tenso_msgs::msg::TensoBlob;

  static void convert_to_ros_message(
    const custom_type & source, ros_message_type & destination)
  {
    // Copy the encoded packet bytes into the message payload. For true
    // zero-copy on the hot path, prefer the intra-process loaned-message path
    // (publish std::move(view)); this conversion is the inter-process fallback.
    destination.payload = source.packet();
  }

  static void convert_to_custom(
    const ros_message_type & source, custom_type & destination)
  {
    destination = tenso_ros::TensorView(source.payload);
  }
};

namespace tenso_ros {

// Convenience alias for declaring pubs/subs.
using TensoTensorAdapted =
  rclcpp::TypeAdapter<tenso_ros::TensorView, tenso_msgs::msg::TensoBlob>;

}  // namespace tenso_ros

#endif  // TENSO_ROS__TYPE_ADAPTER_HPP_
