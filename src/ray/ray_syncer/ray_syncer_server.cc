// Copyright 2024 The Ray Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ray/ray_syncer/ray_syncer_server.h"

#include <chrono>
#include <fstream>
#include <string>
#include <utility>

#include "ray/common/constants.h"

namespace ray::syncer {

// #region agent log
inline void DebugLogServer(const std::string &location,
                           const std::string &message,
                           const std::string &hypothesis_id,
                           const std::string &data = "") {
  static const char *log_path = "/Users/sagar/workspace/codope/ray/.cursor/debug.log";
  auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
  std::ofstream f(log_path, std::ios::app);
  if (f.is_open()) {
    f << "{\"location\":\"" << location << "\",\"message\":\"" << message
      << "\",\"hypothesisId\":\"" << hypothesis_id << "\",\"data\":{" << data
      << "},\"timestamp\":" << now << ",\"sessionId\":\"debug-session\"}\n";
    f.close();
  }
}
// #endregion

namespace {

std::string GetNodeIDFromServerContext(grpc::CallbackServerContext *server_context) {
  const auto &metadata = server_context->client_metadata();
  auto iter = metadata.find("node_id");
  RAY_CHECK(iter != metadata.end());
  return NodeID::FromHex(std::string(iter->second.begin(), iter->second.end())).Binary();
}

}  // namespace

RayServerBidiReactor::RayServerBidiReactor(
    grpc::CallbackServerContext *server_context,
    instrumented_io_context &io_context,
    const std::string &local_node_id,
    std::function<void(std::shared_ptr<const RaySyncMessage>)> message_processor,
    std::function<void(RaySyncerBidiReactor *, bool)> cleanup_cb,
    const std::optional<ray::rpc::AuthenticationToken> &auth_token,
    size_t max_batch_size,
    uint64_t max_batch_delay_ms)
    : RaySyncerBidiReactorBase<ServerBidiReactor>(
          io_context,
          GetNodeIDFromServerContext(server_context),
          std::move(message_processor),
          max_batch_size,
          max_batch_delay_ms),
      cleanup_cb_(std::move(cleanup_cb)),
      server_context_(server_context),
      auth_token_(auth_token) {
  if (auth_token_.has_value() && !auth_token_->empty()) {
    // Validate authentication token
    const auto &metadata = server_context->client_metadata();
    auto it = metadata.find(kAuthTokenKey);
    if (it == metadata.end()) {
      RAY_LOG(WARNING) << "Missing authorization header in syncer connection from node "
                       << NodeID::FromBinary(GetRemoteNodeID());
      Finish(grpc::Status(grpc::StatusCode::UNAUTHENTICATED,
                          "Missing authorization header"));
      return;
    }

    const std::string_view header(it->second.data(), it->second.length());
    ray::rpc::AuthenticationToken provided_token =
        ray::rpc::AuthenticationToken::FromMetadata(header);

    if (!auth_token_->Equals(provided_token)) {
      RAY_LOG(WARNING) << "Invalid bearer token in syncer connection from node "
                       << NodeID::FromBinary(GetRemoteNodeID());
      Finish(grpc::Status(grpc::StatusCode::UNAUTHENTICATED, "Invalid bearer token"));
      return;
    }
  }

  // Send the local node id to the remote
  server_context_->AddInitialMetadata("node_id", NodeID::FromBinary(local_node_id).Hex());
  StartSendInitialMetadata();

  // Start pulling from remote
  StartPull();
}

void RayServerBidiReactor::DoDisconnect() {
  // #region agent log
  DebugLogServer(
      "ray_syncer_server.cc:DoDisconnect",
      "disconnect",
      "B,D",
      "\"this\":\"" + std::to_string(reinterpret_cast<uintptr_t>(this)) + "\"");
  // #endregion
  io_context_.dispatch([this]() { Finish(grpc::Status::OK); }, "");
}

void RayServerBidiReactor::OnCancel() {
  // #region agent log
  DebugLogServer(
      "ray_syncer_server.cc:OnCancel",
      "cancel",
      "B",
      "\"this\":\"" + std::to_string(reinterpret_cast<uintptr_t>(this)) + "\"");
  // #endregion
  io_context_.dispatch([this]() { Disconnect(); }, "");
}

void RayServerBidiReactor::OnDone() {
  // #region agent log
  DebugLogServer("ray_syncer_server.cc:OnDone",
                 "done",
                 "B,D",
                 "\"this\":\"" + std::to_string(reinterpret_cast<uintptr_t>(this)) +
                     "\",\"self_ref_valid\":\"" + (self_ref_ ? "true" : "false") + "\"");
  // #endregion
  io_context_.dispatch(
      [this, cleanup_cb = cleanup_cb_, remote_node_id = GetRemoteNodeID()]() {
        // #region agent log
        DebugLogServer("ray_syncer_server.cc:OnDone:dispatch",
                       "cleanup_executing",
                       "B,E",
                       "\"this\":\"" + std::to_string(reinterpret_cast<uintptr_t>(this)) +
                           "\",\"remote_node_id\":\"" +
                           NodeID::FromBinary(remote_node_id).Hex() + "\"");
        // #endregion
        cleanup_cb(this, false);
        self_ref_.reset();
      },
      "");
}

}  // namespace ray::syncer
