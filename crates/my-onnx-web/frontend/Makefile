# Download/generate external assets for the frontend.
#
# Sources:
#   synset.txt       - ImageNet labels, https://github.com/onnx/models (vision/classification)
#   coco.names       - COCO labels, https://github.com/hunglc007/tensorflow-yolov4-tflite/blob/master/data/classes/coco.names
#   bert-vocab.txt   - BERT base uncased vocab, https://huggingface.co/google-bert/bert-base-uncased
#   gpt2-encoder.json - GPT-2 BPE vocab, https://huggingface.co/openai-community/gpt2
#   gpt2-merges.txt  - GPT-2 BPE merges, https://huggingface.co/openai-community/gpt2

MODELS_DIR := ../../models/validated
PUBLIC := public

ONNX_MODELS_HASH := bec48b6a70e5e9042c0badbaafefe4454e072d08
ONNX_MODELS_REPO := https://github.com/onnx/models
ONNX_CLASSIFICATION := $(ONNX_MODELS_REPO)/raw/$(ONNX_MODELS_HASH)/validated/vision/classification

ASSETS := \
	$(PUBLIC)/synset.txt \
	$(PUBLIC)/coco.names \
	$(PUBLIC)/bert-vocab.txt \
	$(PUBLIC)/gpt2-encoder.json \
	$(PUBLIC)/gpt2-merges.txt

.PHONY: all clean

all: $(ASSETS)

$(PUBLIC)/synset.txt:
	curl -sL "$(ONNX_CLASSIFICATION)/synset.txt" -o $@

$(PUBLIC)/coco.names:
	curl -sL "https://raw.githubusercontent.com/hunglc007/tensorflow-yolov4-tflite/master/data/classes/coco.names" -o $@

$(PUBLIC)/bert-vocab.txt:
	curl -sL "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/vocab.txt" -o $@

$(PUBLIC)/gpt2-encoder.json:
	curl -sL "https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json" -o $@

$(PUBLIC)/gpt2-merges.txt:
	curl -sL "https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt" -o $@

clean:
	rm -f $(ASSETS)
