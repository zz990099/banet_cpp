#include "image_processing_utils/image_processing_utils.hpp"
#include "stereo_banet/banet.hpp"
#include "benchmark_utils/stereo_matching_benchmark_utils.hpp"

using namespace easy_deploy;

#ifdef ENABLE_TENSORRT

#include "trt_core/trt_core.hpp"

std::shared_ptr<BaseStereoMatchingModel> CreateBANetTensorRTModel()
{
  auto engine =
      CreateTrtInferCore("/workspace/models/banet-2d-sceneflow-544-960.engine");
  auto preprocess_block = CreateCudaImageProcessingResizePad(
      ImageProcessingPadMode::TOP_RIGHT, ImageProcessingPadValue::EDGE, true, true,
      {0, 0, 0}, {1, 1, 1});

  return CreateBANetModel(engine, preprocess_block, 544, 960, {"left", "right"},
                                {"disp_pred"});
}

static void benchmark_stereo_matching_banet_tensorrt_sync(benchmark::State &state)
{
  benchmark_stereo_matching_sync(state, CreateBANetTensorRTModel());
}
static void benchmark_stereo_matching_banet_tensorrt_async(benchmark::State &state)
{
  benchmark_stereo_matching_async(state, CreateBANetTensorRTModel());
}
BENCHMARK(benchmark_stereo_matching_banet_tensorrt_sync)->Arg(500)->UseRealTime();
BENCHMARK(benchmark_stereo_matching_banet_tensorrt_async)->Arg(500)->UseRealTime();

#endif

#ifdef ENABLE_ORT

#include "ort_core/ort_core.hpp"

std::shared_ptr<BaseStereoMatchingModel> CreateBANetOnnxRuntimeModel()
{
  auto engine =
      CreateOrtInferCore("/workspace/models/banet-2d-sceneflow-544-960.onnx",
                         {{"left", {1, 3, 544, 960}}, {"right", {1, 3, 544, 960}}},
                         {{"disp_pred", {1, 1, 544, 960}}});
  auto preprocess_block = CreateCpuImageProcessingResizePad(
      ImageProcessingPadMode::TOP_RIGHT, ImageProcessingPadValue::EDGE, true, true,
      {0, 0, 0}, {1, 1, 1});

  return CreateBANetModel(engine, preprocess_block, 544, 960, {"left", "right"},
                                {"disp_pred"});
}

static void benchmark_stereo_matching_banet_onnxruntime_sync(benchmark::State &state)
{
  benchmark_stereo_matching_sync(state, CreateBANetOnnxRuntimeModel());
}
static void benchmark_stereo_matching_banet_onnxruntime_async(benchmark::State &state)
{
  benchmark_stereo_matching_async(state, CreateBANetOnnxRuntimeModel());
}
BENCHMARK(benchmark_stereo_matching_banet_onnxruntime_sync)->Arg(30)->UseRealTime();
BENCHMARK(benchmark_stereo_matching_banet_onnxruntime_async)->Arg(30)->UseRealTime();

#endif

#ifdef ENABLE_RKNN

#include "rknn_core/rknn_core.hpp"

std::shared_ptr<BaseStereoMatchingModel> CreateBANetRknnModel()
{
  auto engine = CreateRknnInferCore(
      "/workspace/models/banet-2d-sceneflow-544-960.rknn",
      {{"left", RknnInputTensorType::RK_UINT8}, {"right", RknnInputTensorType::RK_UINT8}},
      5, 3);
  auto preprocess_block = CreateCpuImageProcessingResizePad(
      ImageProcessingPadMode::TOP_RIGHT, ImageProcessingPadValue::EDGE, false, false, {}, {});

  return CreateBANetModel(engine, preprocess_block, 544, 960, {"left", "right"},
                                {"disp_pred"});
}

static void benchmark_stereo_matching_banet_rknn_sync(benchmark::State &state)
{
  benchmark_stereo_matching_sync(state, CreateBANetRknnModel());
}
static void benchmark_stereo_matching_banet_rknn_async(benchmark::State &state)
{
  benchmark_stereo_matching_async(state, CreateBANetRknnModel());
}
BENCHMARK(benchmark_stereo_matching_banet_rknn_sync)->Arg(100)->UseRealTime();
BENCHMARK(benchmark_stereo_matching_banet_rknn_async)->Arg(200)->UseRealTime();

#endif

BENCHMARK_MAIN();