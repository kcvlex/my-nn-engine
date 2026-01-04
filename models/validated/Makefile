WGET = wget
HASH = bec48b6a70e5e9042c0badbaafefe4454e072d08
REPO = https://github.com/onnx/models
CLASSIFICATION = $(REPO)/raw/$(HASH)/validated/vision/classification
OBJECT_DETECTION_SEGMENTATION = $(REPO)/raw/$(HASH)/validated/vision/object_detection_segmentation
TEXT = $(REPO)/raw/$(HASH)/validated/text/machine_comprehension

.PHONY: all
all: mnist-12 resnet18-v2-7 resnet152-v2-7 bertsquad-12 yolov4 gpt2-10

mnist-12: mnist-12.tar.gz
	tar -xvf $<

mnist-12.tar.gz:
	$(WGET) -O $@ $(CLASSIFICATION)/mnist/model/$@

resnet18-v2-7: resnet18-v2-7.tar.gz
	tar -xvf $<

resnet18-v2-7.tar.gz:
	$(WGET) -O $@ $(CLASSIFICATION)/resnet/model/$@

resnet152-v2-7: resnet152-v2-7.tar.gz
	tar -xvf $<

resnet152-v2-7.tar.gz:
	$(WGET) -O $@ $(CLASSIFICATION)/resnet/model/$@

bertsquad-12: bertsquad-12.tar.gz
	tar -xvf $<

bertsquad-12.tar.gz:
	$(WGET) -O $@ $(TEXT)/bert-squad/model/$@

yolov4: yolov4.tar.gz
	tar -xvf $<

yolov4.tar.gz:
	$(WGET) -O $@ $(OBJECT_DETECTION_SEGMENTATION)/yolov4/model/$@

gpt2-10: gpt2-10.tar.gz
	tar -xvf $<

gpt2-10.tar.gz:
	$(WGET) -O $@ $(TEXT)/gpt-2/model/$@
