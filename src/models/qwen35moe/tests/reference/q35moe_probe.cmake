# The C++ reference probe of gufo-qwen35moe (q35moe_probe.cpp), as a test
# target of the extension. Copy this file and q35moe_probe.cpp into
# $EXTENSION/tests/ and apply q35moe-probe-hook.patch (README.md). Built on
# request only:
#   cmake --build build/q35moe-gpu --target q35moe_probe
gufo_extension_target(
  NAME q35moe_probe
  SOURCES tests/q35moe_probe.cpp
  LINK_LIBRARIES
    strycast_q35_hip
    strycast_q35
    strycast_q35_kernels
    strycast_q35_family
    gufo_ext_hip
    gufo_ext_cache
    gufo_ext_core
    gufo_ext_runtime
)
if(TARGET q35moe_probe)
  set_target_properties(q35moe_probe PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()
