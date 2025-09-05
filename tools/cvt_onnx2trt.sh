#!/bin/bash

echo "Converting BANet onnx model to tensorrt ..."
/usr/src/tensorrt/bin/trtexec --onnx=/workspace/models/banet-2d-sceneflow-544-960.onnx \
                              --saveEngine=/workspace/models/banet-2d-sceneflow-544-960.engine \
                              --fp16
