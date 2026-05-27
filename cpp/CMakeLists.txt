set(AZ_CORE_SOURCES
  src/board.cpp
  src/movegen.cpp
  src/magic.cpp
  src/encoding.cpp
  src/node.cpp
  src/inference_queue.cpp
  src/mcts.cpp
  src/selfplay.cpp
)

pybind11_add_module(_az_core src/bindings.cpp ${AZ_CORE_SOURCES})

target_include_directories(_az_core PRIVATE include)

if(NOT MSVC)
  target_compile_options(_az_core PRIVATE -O3)
endif()

install(TARGETS _az_core LIBRARY DESTINATION . RUNTIME DESTINATION .)

# Copy extension beside Python package for editable installs
add_custom_command(
  TARGET _az_core POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:_az_core>
          ${CMAKE_SOURCE_DIR}/python/az/
  COMMENT "Copy _az_core into python/az for editable install")

# Perft test executable
add_executable(test_perft tests/test_perft.cpp ${AZ_CORE_SOURCES})
target_include_directories(test_perft PRIVATE include)
if(NOT MSVC)
  target_compile_options(test_perft PRIVATE -O3)
endif()

enable_testing()
add_test(NAME perft COMMAND test_perft)
