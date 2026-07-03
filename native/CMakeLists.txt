cmake_minimum_required(VERSION 3.24)

project(style_bert_vits2_ggml_onnx_ep LANGUAGES CXX)

set(ORT_INCLUDE_DIR "" CACHE PATH "Path to ONNX Runtime public headers")

if(NOT ORT_INCLUDE_DIR)
  message(FATAL_ERROR "Set ORT_INCLUDE_DIR to the directory containing onnxruntime_cxx_api.h")
endif()

add_library(style_bert_vits2_ggml_onnx_ep SHARED
  src/style_bert_vits2_ggml_onnx_ep.cc
)

target_compile_features(style_bert_vits2_ggml_onnx_ep PRIVATE cxx_std_17)
target_include_directories(style_bert_vits2_ggml_onnx_ep PRIVATE "${ORT_INCLUDE_DIR}")
target_link_libraries(style_bert_vits2_ggml_onnx_ep PRIVATE ${CMAKE_DL_LIBS})
target_compile_definitions(style_bert_vits2_ggml_onnx_ep PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)

set_target_properties(style_bert_vits2_ggml_onnx_ep PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  OUTPUT_NAME style_bert_vits2_ggml_onnx_ep
)

install(TARGETS style_bert_vits2_ggml_onnx_ep
  LIBRARY DESTINATION onnxruntime_ep_style_bert_vits2_ggml/lib
  RUNTIME DESTINATION onnxruntime_ep_style_bert_vits2_ggml/lib
)
