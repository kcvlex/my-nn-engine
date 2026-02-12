WGET = wget
HASH = bec48b6a70e5e9042c0badbaafefe4454e072d08
REPO = https://github.com/onnx/models
CLASSIFICATION = $(REPO)/raw/$(HASH)/validated/vision/classification
OBJECT_DETECTION_SEGMENTATION = $(REPO)/raw/$(HASH)/validated/vision/object_detection_segmentation
TEXT = $(REPO)/raw/$(HASH)/validated/text/machine_comprehension

# SHA256 checksums for model files
MNIST_12_SHA256 = a53a59dcaca8804a0f6dfda9a3cf2e082979589391dbde73640b60684f1d24e9
RESNET18_V2_7_SHA256 = a64c111fac5d4e8bebf9246507ec7a92ea1c1596de09fca73bb7f94e9750e316
RESNET152_V2_7_SHA256 = 78b507483050a427870a264c398d121f44b92d69c8ca0c53217eb5137b6b51e3
BERTSQUAD_12_SHA256 = 4cd041010ab4ad11c23c0eb7f056c4e4286a894b6d712ef09445b955763fb1b1
YOLOV4_SHA256 = 7d06db57b6b4fff4f75dd3bd16f29b32faec4552d763869d5a3c37d25542cbb2
GPT2_10_SHA256 = 0c9b81350c95cf2f40e8e72f76792ef6b3c73894beee571171739079330aad53

.PHONY: all
all: mnist-12 resnet18-v2-7 resnet152-v2-7 bertsquad-12 yolov4 gpt2-10

mnist-12: mnist-12.tar.gz
	echo "$(MNIST_12_SHA256)  $<" | sha256sum -c -
	tar -xvf $<

mnist-12.tar.gz:
	$(WGET) -O $@ $(CLASSIFICATION)/mnist/model/$@

resnet18-v2-7: resnet18-v2-7.tar.gz
	echo "$(RESNET18_V2_7_SHA256)  $<" | sha256sum -c -
	tar -xvf $<

resnet18-v2-7.tar.gz:
	$(WGET) -O $@ $(CLASSIFICATION)/resnet/model/$@

resnet152-v2-7: resnet152-v2-7.tar.gz
	echo "$(RESNET152_V2_7_SHA256)  $<" | sha256sum -c -
	tar -xvf $<

resnet152-v2-7.tar.gz:
	$(WGET) -O $@ $(CLASSIFICATION)/resnet/model/$@

bertsquad-12: bertsquad-12.tar.gz
	echo "$(BERTSQUAD_12_SHA256)  $<" | sha256sum -c -
	tar -xvf $<

bertsquad-12.tar.gz:
	$(WGET) -O $@ $(TEXT)/bert-squad/model/$@

yolov4: yolov4.tar.gz
	echo "$(YOLOV4_SHA256)  $<" | sha256sum -c -
	tar -xvf $<

yolov4.tar.gz:
	$(WGET) -O $@ $(OBJECT_DETECTION_SEGMENTATION)/yolov4/model/$@

gpt2-10: gpt2-10.tar.gz
	echo "$(GPT2_10_SHA256)  $<" | sha256sum -c -
	tar -xvf $<

gpt2-10.tar.gz:
	$(WGET) -O $@ $(TEXT)/gpt-2/model/$@
