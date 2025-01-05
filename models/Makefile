WGET = wget
HASH = bec48b6a70e5e9042c0badbaafefe4454e072d08
REPO = https://github.com/onnx/models
CLASSIFICATION = $(REPO)/raw/$(HASH)/validated/vision/classification

mnist-12.onnx:
	$(WGET) -O $@ $(CLASSIFICATION)/mnist/model/mnist-12.onnx

resnet152-v2-7.onnx:
	$(WGET) -O $@ $(CLASSIFICATION)/resnet/model/resnet152-v2-7.onnx

resnet18-v2-7.onnx:
	$(WGET) -O $@ $(CLASSIFICATION)/resnet/model/resnet18-v2-7.onnx

mobilenetv2-12.onnx:
	$(WGET) -O $@ $(CLASSIFICATION)/mobilenet/model/mobilenetv2-12.onnx
