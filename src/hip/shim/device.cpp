// C ABI over hipGetDeviceProperties and hipPointerGetAttributes
// (docs/RUST.md).
//
// hipDeviceProp_t and hipPointerAttribute_t change their layouts (and
// hipMemoryType its values) between ROCm releases, so the compiler reads them
// against the installed headers here and copies the fields gufo reports into
// plain C types.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

extern "C" {

struct GufoDeviceProperties {
  char name[256];
  char arch_name[256];
  std::int32_t major;
  std::int32_t minor;
  std::uint64_t total_global_memory;
  std::int32_t multiprocessor_count;
  std::int32_t warp_size;
  std::int32_t integrated;
};

// Returns the hipError_t of hipGetDeviceProperties; `out` is filled only on
// success, with NUL-terminated names.
std::int32_t gufo_hip_device_properties(std::int32_t device,
                                        GufoDeviceProperties* out) noexcept {
  hipDeviceProp_t properties{};
  const hipError_t status = hipGetDeviceProperties(&properties, device);
  if (status != hipSuccess) {
    return static_cast<std::int32_t>(status);
  }
  static_assert(sizeof(properties.name) <= sizeof(out->name));
  static_assert(sizeof(properties.gcnArchName) <= sizeof(out->arch_name));
  std::memset(out, 0, sizeof(*out));
  std::memcpy(out->name, properties.name, sizeof(properties.name));
  std::memcpy(out->arch_name, properties.gcnArchName,
              sizeof(properties.gcnArchName));
  out->name[sizeof(out->name) - 1] = '\0';
  out->arch_name[sizeof(out->arch_name) - 1] = '\0';
  out->major = properties.major;
  out->minor = properties.minor;
  out->total_global_memory = properties.totalGlobalMem;
  out->multiprocessor_count = properties.multiProcessorCount;
  out->warp_size = properties.warpSize;
  out->integrated = properties.integrated;
  return static_cast<std::int32_t>(hipSuccess);
}

// The memory types gufo names, independent of the hipMemoryType values.
enum GufoMemoryType : std::int32_t {
  kGufoMemoryUnregistered = 0,
  kGufoMemoryHost = 1,
  kGufoMemoryDevice = 2,
  kGufoMemoryManaged = 3,
  kGufoMemoryArray = 4,
  kGufoMemoryUnified = 5,
  kGufoMemoryUnknown = 6,
};

// Returns the hipError_t of hipPointerGetAttributes; `type` and `managed` are
// set only on success.
std::int32_t gufo_hip_pointer_attributes(const void* pointer,
                                         std::int32_t* type,
                                         std::int32_t* managed) noexcept {
  hipPointerAttribute_t attributes{};
  const hipError_t status = hipPointerGetAttributes(&attributes, pointer);
  if (status != hipSuccess) {
    return static_cast<std::int32_t>(status);
  }
  switch (attributes.type) {
    case hipMemoryTypeUnregistered:
      *type = kGufoMemoryUnregistered;
      break;
    case hipMemoryTypeHost:
      *type = kGufoMemoryHost;
      break;
    case hipMemoryTypeDevice:
      *type = kGufoMemoryDevice;
      break;
    case hipMemoryTypeManaged:
      *type = kGufoMemoryManaged;
      break;
    case hipMemoryTypeArray:
      *type = kGufoMemoryArray;
      break;
    case hipMemoryTypeUnified:
      *type = kGufoMemoryUnified;
      break;
    default:
      *type = kGufoMemoryUnknown;
      break;
  }
  *managed = attributes.isManaged != 0 ? 1 : 0;
  return static_cast<std::int32_t>(hipSuccess);
}

}  // extern "C"
