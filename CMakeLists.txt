cmake_minimum_required(VERSION 3.16)
project(policy_runner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Booster Robotics SDK
# Prefer an explicit root, otherwise fall back to a system install
# (sudo ./install.sh from booster_robotics_sdk copies into /usr/local).
#
#   cmake .. -DBOOSTER_SDK_DIR=/path/to/booster_robotics_sdk
# ---------------------------------------------------------------------------
set(BOOSTER_SDK_DIR "" CACHE PATH "Path to booster_robotics_sdk root (include/ + lib/)")

find_package(Threads REQUIRED)

set(_booster_arch "${CMAKE_SYSTEM_PROCESSOR}")
if(_booster_arch MATCHES "^(AMD64|amd64)$")
  set(_booster_arch x86_64)
endif()

set(_booster_include_hints
  /usr/local/include
  /usr/include
)
set(_booster_lib_hints
  /usr/local/lib
  /usr/lib
)

if(BOOSTER_SDK_DIR)
  list(PREPEND _booster_include_hints "${BOOSTER_SDK_DIR}/include")
  list(PREPEND _booster_lib_hints
    "${BOOSTER_SDK_DIR}/lib/${_booster_arch}"
    "${BOOSTER_SDK_DIR}/lib/${_booster_arch}/22"
    "${BOOSTER_SDK_DIR}/lib/${_booster_arch}/24"
    "${BOOSTER_SDK_DIR}/lib"
  )
endif()

find_path(BOOSTER_SDK_INCLUDE_DIR
  NAMES booster/idl/b1/LowCmd.h
  HINTS ${_booster_include_hints}
  NO_DEFAULT_PATH
)
if(NOT BOOSTER_SDK_INCLUDE_DIR)
  find_path(BOOSTER_SDK_INCLUDE_DIR NAMES booster/idl/b1/LowCmd.h)
endif()

find_library(BOOSTER_SDK_LIBRARY
  NAMES booster_robotics_sdk
  HINTS ${_booster_lib_hints}
  NO_DEFAULT_PATH
)
if(NOT BOOSTER_SDK_LIBRARY)
  find_library(BOOSTER_SDK_LIBRARY NAMES booster_robotics_sdk)
endif()

if(NOT BOOSTER_SDK_INCLUDE_DIR OR NOT BOOSTER_SDK_LIBRARY)
  message(FATAL_ERROR
    "Booster Robotics SDK not found.\n"
    "  Install it (sudo ./install.sh in booster_robotics_sdk), or pass:\n"
    "    cmake .. -DBOOSTER_SDK_DIR=/path/to/booster_robotics_sdk\n"
    "  Looking for: booster/idl/b1/LowCmd.h and libbooster_robotics_sdk")
endif()

message(STATUS "Booster SDK include: ${BOOSTER_SDK_INCLUDE_DIR}")
message(STATUS "Booster SDK library: ${BOOSTER_SDK_LIBRARY}")

# FastDDS pieces shipped with the SDK (needed when linking the static SDK lib).
find_library(FASTRTPS_LIBRARY NAMES fastrtps HINTS ${_booster_lib_hints})
find_library(FASTCDR_LIBRARY NAMES fastcdr HINTS ${_booster_lib_hints})
find_library(FOONATHAN_MEMORY_LIBRARY
  NAMES foonathan_memory-0.7.3 foonathan_memory
  HINTS ${_booster_lib_hints}
)

add_executable(policy_runner
  src/policy_runner.cpp
  src/robot/robot_bridge.cpp
  src/policy/sine_arm_policy.cpp
  src/policy/step_arm_policy.cpp
  src/policy/sine_knee_policy.cpp
  src/policy/hold_lower_body_policy.cpp
)

target_include_directories(policy_runner PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/include
  ${BOOSTER_SDK_INCLUDE_DIR}
)

target_link_libraries(policy_runner PRIVATE
  ${BOOSTER_SDK_LIBRARY}
  Threads::Threads
  ${CMAKE_DL_LIBS}
)

if(FASTRTPS_LIBRARY)
  target_link_libraries(policy_runner PRIVATE ${FASTRTPS_LIBRARY})
endif()
if(FASTCDR_LIBRARY)
  target_link_libraries(policy_runner PRIVATE ${FASTCDR_LIBRARY})
endif()
if(FOONATHAN_MEMORY_LIBRARY)
  target_link_libraries(policy_runner PRIVATE ${FOONATHAN_MEMORY_LIBRARY})
endif()

if(UNIX AND NOT APPLE)
  target_link_libraries(policy_runner PRIVATE rt)
endif()
