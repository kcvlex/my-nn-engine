WGET = wget
HASH = bec48b6a70e5e9042c0badbaafefe4454e072d08
REPO = https://github.com/onnx/models
CLASSIFICATION = $(REPO)/raw/$(HASH)/validated/vision/classification

.PHONY: mnist-12
mnist-12: mnist-12.tar.gz
	tar -xvf $<

mnist-12.tar.gz:
	$(WGET) -O $@ $(CLASSIFICATION)/mnist/model/$@

.PHONY: resnet18-v2-7
resnet18-v2-7: resnet18-v2-7.tar.gz
	tar -xvf $<

resnet18-v2-7.tar.gz:
	$(WGET) -O $@ $(CLASSIFICATION)/resnet/model/$@
