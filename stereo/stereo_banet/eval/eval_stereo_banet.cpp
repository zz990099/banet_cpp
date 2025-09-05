#include "image_processing_utils/image_processing_utils.hpp"
#include "stereo_banet/banet.hpp"
#include "eval_utils/stereo_matching_eval_utils.hpp"

using namespace easy_deploy;

#ifdef ENABLE_TENSORRT

#include "trt_core/trt_core.hpp"

class EvalAccuracyBANetTensorRTFixture : public EvalAccuracyStereoMatchingFixture {
public:
  SetUpReturnType SetUp() override
  {
    auto engine = CreateTrtInferCore("/workspace/models/banet-2d-sceneflow-544-960.engine");
    auto preprocess_block = CreateCudaImageProcessingResizePad(ImageProcessingPadMode::TOP_RIGHT,
                                                               ImageProcessingPadValue::EDGE, true,
                                                               true, {0, 0, 0}, {1, 1, 1});

    auto model =
        CreateBANetModel(engine, preprocess_block, 544, 960, {"left", "right"}, {"disp_pred"});

    const std::string sceneflow_val_txt_path =
        "/workspace/test_data/sceneflow/sceneflow_finalpass_test.txt";
    return {model, sceneflow_val_txt_path};
  }
};

RegisterEvalAccuracyStereoMatching(EvalAccuracyBANetTensorRTFixture);

#endif

#ifdef ENABLE_ORT

#include "ort_core/ort_core.hpp"

class EvalAccuracyBANetOnnxRuntimeFixture : public EvalAccuracyStereoMatchingFixture {
public:
  SetUpReturnType SetUp() override
  {
    auto engine =
        CreateOrtInferCore("/workspace/models/banet-2d-sceneflow-544-960.onnx",
                           {{"left", {1, 3, 544, 960}}, {"right", {1, 3, 544, 960}}},
                           {{"disp_pred", {1, 1, 544, 960}}});
    auto preprocess_block = CreateCpuImageProcessingResizePad(ImageProcessingPadMode::LETTER_BOX,
                                                              ImageProcessingPadValue::EDGE, true,
                                                              true, {0, 0, 0}, {1, 1, 1});

    auto model =
        CreateBANetModel(engine, preprocess_block, 544, 960, {"left", "right"}, {"disp_pred"});

    const std::string sceneflow_val_txt_path =
        "/workspace/test_data/sceneflow/sceneflow_finalpass_test.txt";
    return {model, sceneflow_val_txt_path};
  }
};

RegisterEvalAccuracyStereoMatching(EvalAccuracyBANetOnnxRuntimeFixture);

#endif

#ifdef ENABLE_RKNN

#include "rknn_core/rknn_core.hpp"

class EvalAccuracyBANetRknnFixture : public EvalAccuracyStereoMatchingFixture {
public:
  SetUpReturnType SetUp() override
  {
    auto engine = CreateRknnInferCore(
        "/workspace/models/banet-2d-sceneflow-544-960.rknn",
        {{"left", RknnInputTensorType::RK_UINT8}, {"right", RknnInputTensorType::RK_UINT8}}, 5, 3);
    auto preprocess_block = CreateCpuImageProcessingResizePad(
        ImageProcessingPadMode::TOP_RIGHT, ImageProcessingPadValue::EDGE, false, false, {}, {});

    auto model =
        CreateBANetModel(engine, preprocess_block, 544, 960, {"left", "right"}, {"disp_pred"});

    const std::string sceneflow_val_txt_path =
        "/workspace/test_data/sceneflow/sceneflow_finalpass_test.txt";
    return {model, sceneflow_val_txt_path};
  }
};

RegisterEvalAccuracyStereoMatching(EvalAccuracyBANetRknnFixture);

#endif

EVAL_MAIN()
