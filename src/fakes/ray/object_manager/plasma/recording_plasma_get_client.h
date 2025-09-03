// Copyright 2025 The Ray Authors.
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

#pragma once

#include <memory>
#include <vector>

#include "ray/common/buffer.h"
#include "ray/common/id.h"
#include "ray/common/status.h"
#include "ray/object_manager/plasma/client.h"

namespace ray {
namespace fakes {

// Wrapper over PlasmaClientInterface that records Get batches.
class RecordingPlasmaGetClient : public plasma::PlasmaClientInterface {
 public:
  RecordingPlasmaGetClient(std::shared_ptr<plasma::PlasmaClientInterface> inner,
                           std::vector<std::vector<ObjectID>> *observed)
      : inner_(std::move(inner)), observed_(observed) {}

  Status Connect(const std::string &a, const std::string &b, int c) override {
    return inner_->Connect(a, b, c);
  }
  Status Release(const ObjectID &id) override { return inner_->Release(id); }
  Status Contains(const ObjectID &id, bool *has) override {
    return inner_->Contains(id, has);
  }
  Status Disconnect() override { return inner_->Disconnect(); }

  Status Get(const std::vector<ObjectID> &object_ids,
             int64_t timeout_ms,
             std::vector<plasma::ObjectBuffer> *object_buffers) override {
    if (observed_ != nullptr) {
      observed_->push_back(object_ids);
    }
    // Return non-null buffers to simulate presence for tests.
    object_buffers->resize(object_ids.size());
    for (size_t i = 0; i < object_ids.size(); i++) {
      uint8_t byte = 0;
      auto parent = std::make_shared<LocalMemoryBuffer>(&byte, 1, /*copy_data=*/true);
      (*object_buffers)[i].data = SharedMemoryBuffer::Slice(parent, 0, 1);
      (*object_buffers)[i].metadata = SharedMemoryBuffer::Slice(parent, 0, 1);
    }
    return Status::OK();
  }

  Status GetExperimentalMutableObject(
      const ObjectID &id, std::unique_ptr<plasma::MutableObject> *obj) override {
    return inner_->GetExperimentalMutableObject(id, obj);
  }
  Status Seal(const ObjectID &id) override { return inner_->Seal(id); }
  Status Abort(const ObjectID &id) override { return inner_->Abort(id); }
  Status CreateAndSpillIfNeeded(const ObjectID &id,
                                const rpc::Address &addr,
                                bool is_mutable,
                                int64_t data_size,
                                const uint8_t *metadata,
                                int64_t metadata_size,
                                std::shared_ptr<Buffer> *data,
                                plasma::flatbuf::ObjectSource source,
                                int device_num) override {
    return inner_->CreateAndSpillIfNeeded(id,
                                          addr,
                                          is_mutable,
                                          data_size,
                                          metadata,
                                          metadata_size,
                                          data,
                                          source,
                                          device_num);
  }
  Status TryCreateImmediately(const ObjectID &id,
                              const rpc::Address &addr,
                              int64_t data_size,
                              const uint8_t *metadata,
                              int64_t metadata_size,
                              std::shared_ptr<Buffer> *data,
                              plasma::flatbuf::ObjectSource source,
                              int device_num) override {
    return inner_->TryCreateImmediately(
        id, addr, data_size, metadata, metadata_size, data, source, device_num);
  }
  Status Delete(const std::vector<ObjectID> &ids) override { return inner_->Delete(ids); }

 private:
  std::shared_ptr<plasma::PlasmaClientInterface> inner_;
  std::vector<std::vector<ObjectID>> *observed_;
};

}  // namespace fakes
}  // namespace ray
