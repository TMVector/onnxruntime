// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <map>
#include <string>
#include <cstring>
#include <type_traits>

#include "core/common/common.h"
#include "core/common/exceptions.h"
#include "core/common/status.h"
#include "core/framework/fence.h"
#include "core/session/onnxruntime_c_api.h"

// Struct to represent a physical device.
struct OrtDevice {
  using DeviceType = int8_t;
  using MemoryType = int8_t;
  using DeviceId = int16_t;

  // Pre-defined device types.
  static const DeviceType CPU = 0;
  static const DeviceType GPU = 1;  //CUDA
  static const DeviceType FPGA = 2;

  struct MemType {
    // Pre-defined memory types.
    static const MemoryType DEFAULT = 0;
    static const MemoryType CUDA_PINNED = 1;
  };

  constexpr OrtDevice(DeviceType device_type_, MemoryType memory_type_, DeviceId device_id_)
      : device_type(device_type_),
        memory_type(memory_type_),
        device_id(device_id_) {}

  constexpr OrtDevice() : OrtDevice(CPU, MemType::DEFAULT, 0) {}

  DeviceType Type() const {
    return device_type;
  }

  MemoryType MemType() const {
    return memory_type;
  }

  DeviceId Id() const {
    return device_id;
  }

  std::string ToString() const {
    std::ostringstream ostr;
    ostr << "Device: ["
         << " type:" << static_cast<int>(device_type)
         << " memory_type:" << static_cast<int>(memory_type)
         << " device_id:" << device_id
         << "]";
    return ostr.str();
  }

 private:
  // Device type.
  DeviceType device_type;

  // Memory type.
  MemoryType memory_type;

  // Device index.
  DeviceId device_id;
};

inline bool operator==(const OrtDevice& left, const OrtDevice& other) {
  return left.Id() == other.Id() && left.MemType() == other.MemType() && left.Type() == other.Type();
}

inline bool operator!=(const OrtDevice& left, const OrtDevice& other) {
  return !(left == other);
}

struct OrtMemoryInfo {
  // use string for name, so we could have customized allocator in execution provider.
  const char* name;
  int id;
  OrtMemType mem_type;
  OrtAllocatorType type;
  OrtDevice device;

  constexpr OrtMemoryInfo(const char* name_, OrtAllocatorType type_, OrtDevice device_ = OrtDevice(), int id_ = 0, OrtMemType mem_type_ = OrtMemTypeDefault)
#if (defined(__GNUC__) || defined(__clang__))
      __attribute__((nonnull))
#endif
      : name(name_),
        id(id_),
        mem_type(mem_type_),
        type(type_),
        device(device_) {
  }

  // To make OrtMemoryInfo become a valid key in std map
  inline bool operator<(const OrtMemoryInfo& other) const {
    if (type != other.type)
      return type < other.type;
    if (mem_type != other.mem_type)
      return mem_type < other.mem_type;
    if (id != other.id)
      return id < other.id;

    return strcmp(name, other.name) < 0;
  }

  inline std::string ToString() const {
    std::ostringstream ostr;
    ostr << "OrtMemoryInfo: ["
         << " name:" << name
         << " id:" << id
         << " mem_type:" << mem_type
         << " type:" << type
         << "]";
    return ostr.str();
  }
};

inline bool operator==(const OrtMemoryInfo& left, const OrtMemoryInfo& other) {
  return left.mem_type == other.mem_type && left.type == other.type && left.id == other.id &&
         strcmp(left.name, other.name) == 0;
}

inline bool operator!=(const OrtMemoryInfo& lhs, const OrtMemoryInfo& rhs) { return !(lhs == rhs); }

std::ostream& operator<<(std::ostream& out, const OrtMemoryInfo& info);

namespace onnxruntime {
constexpr const char* CPU = "Cpu";
constexpr const char* CUDA = "Cuda";
constexpr const char* CUDA_PINNED = "CudaPinned";
constexpr const char* TRT = "Tensorrt";
constexpr const char* TRT_PINNED = "TensorrtPinned";

// forward declaration
class SessionState;

template <typename T>
using IAllocatorUniquePtr = std::unique_ptr<T, std::function<void(T*)>>;

class IAllocator {
 public:
  virtual ~IAllocator() = default;
  virtual void* Alloc(size_t size) = 0;
  virtual void Free(void* p) = 0;
  virtual const OrtMemoryInfo& Info() const = 0;

  /**
     optional CreateFence interface, as provider like DML has its own fence
  */
  virtual FencePtr CreateFence(const SessionState* /*unused*/) { return nullptr; }

  static bool CalcMemSizeForArray(size_t nmemb, size_t size, size_t* out) noexcept {
    return CalcMemSizeForArrayWithAlignment<0>(nmemb, size, out);
  }

  /**
   * https://cwe.mitre.org/data/definitions/190.html
   * \tparam alignment must be power of 2
   * \param nmemb
   * \param size
   * \param out
   * \return true, successful. false, overflow
   */
  template <size_t alignment>
  static bool CalcMemSizeForArrayWithAlignment(size_t nmemb, size_t size, size_t* out) noexcept ORT_MUST_USE_RESULT;
  /**
   * allocate memory for an array which has nmemb items of data, each size bytes long
   */
  void* AllocArray(size_t nmemb, size_t size) {
    size_t len;
    if (!CalcMemSizeForArray(nmemb, size, &len))
      return nullptr;
    return Alloc(len);
  }

  /**
 * allocate memory for an array which has nmemb items of data, each size bytes long
 */
  template <size_t alignment>
  void* AllocArrayWithAlignment(size_t nmemb, size_t size) {
    size_t len;
    if (!CalcMemSizeForArrayWithAlignment<alignment>(nmemb, size, &len))
      return nullptr;
    return Alloc(len);
  }

  /**
     Create a std::unique_ptr that is allocated and freed by the provided IAllocator.
     @param allocator The allocator.
     @param count_or_bytes The exact bytes to allocate if T is void, otherwise the number of elements to allocate.
     @returns std::unique_ptr with allocated memory and deleter.
  */
  template <typename T>
  static IAllocatorUniquePtr<T> MakeUniquePtr(std::shared_ptr<IAllocator> allocator, size_t count_or_bytes) {
    if (allocator == nullptr) return nullptr;
    // for now limit to fundamental types. we could support others, but to do so either we or the caller
    // needs to call the dtor for the objects, for buffers allocated on device we don't have destructor
    //static_assert(std::is_fundamental<T>::value, "Fundamental type required as no destructors are called.");

    size_t alloc_size = count_or_bytes;

    // if T is not void, 'count_or_bytes' == number of items so allow for that
    if (!std::is_void<T>::value) {
      // sizeof(void) isn't valid, but the compiler isn't smart enough to ignore that this line isn't
      // reachable if T is void. use std::conditional to 'use' void* in the sizeof call
      if (!CalcMemSizeForArray(count_or_bytes, sizeof(typename std::conditional<std::is_void<T>::value, void*, T>::type),
                               &alloc_size)) return nullptr;
    }
    return IAllocatorUniquePtr<T>{
        static_cast<T*>(allocator->Alloc(alloc_size)),  // allocate
        [=](T* ptr) { allocator->Free(ptr); }};         // capture IAllocator so it's always valid, and use as deleter
  }
};

template <size_t alignment>
bool IAllocator::CalcMemSizeForArrayWithAlignment(size_t nmemb, size_t size, size_t* out) noexcept {
  static constexpr size_t max_allowed = (static_cast<size_t>(1) << (static_cast<size_t>(std::numeric_limits<size_t>::digits >> 1))) - alignment;
  static constexpr size_t max_size = std::numeric_limits<size_t>::max() - alignment;
  static constexpr size_t alignment_mask = alignment - 1;
  //Indeed, we only need to check if max_size / nmemb < size
  //max_allowed is for avoiding unnecessary DIV.
  if (nmemb >= max_allowed && max_size / nmemb < size) {
    return false;
  }
  if (size >= max_allowed &&
      nmemb > 0 && max_size / nmemb < size) {
    return false;
  }
  if (alignment == 0)
    *out = size * nmemb;
  else
    *out = (size * nmemb + alignment_mask) & ~static_cast<size_t>(alignment_mask);
  return true;
}

/**
   The resource allocator on a physical device.
   This allocator will directly allocate resource from system call
*/
class IDeviceAllocator : public IAllocator {
 public:
  ~IDeviceAllocator() override = default;
  void* Alloc(size_t size) override = 0;
  void Free(void* p) override = 0;
  const OrtMemoryInfo& Info() const override = 0;
  virtual bool AllowsArena() const { return true; }
};

class CPUAllocator : public IDeviceAllocator {
 public:
  explicit CPUAllocator(std::unique_ptr<OrtMemoryInfo> memory_info) {
    ORT_ENFORCE(nullptr != memory_info);
    memory_info_ = std::move(memory_info);
  }

  CPUAllocator() {
    memory_info_ = std::make_unique<OrtMemoryInfo>(CPU, OrtAllocatorType::OrtDeviceAllocator);
  }

  void* Alloc(size_t size) override;
  void Free(void* p) override;
  const OrtMemoryInfo& Info() const override;

 private:
  std::unique_ptr<OrtMemoryInfo> memory_info_;
};

using AllocatorPtr = std::shared_ptr<IAllocator>;

}  // namespace onnxruntime
