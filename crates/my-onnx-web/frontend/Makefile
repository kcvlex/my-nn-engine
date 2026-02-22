# Download/generate external assets for the frontend.
#
# Sources:
#   synset.txt       - ImageNet labels, copied from models/validated/synset.txt
#   coco.names       - COCO labels, https://github.com/hunglc007/tensorflow-yolov4-tflite/blob/master/data/classes/coco.names
#   bert-vocab.txt   - BERT base uncased vocab, https://huggingface.co/google-bert/bert-base-uncased
#   gpt2-encoder.json - GPT-2 BPE vocab, https://huggingface.co/openai-community/gpt2
#   gpt2-merges.txt  - GPT-2 BPE merges, https://huggingface.co/openai-community/gpt2
#   gpt2-wte-f16.bin - GPT-2 word token embedding (float16), extracted from the ONNX model

MODELS_DIR := ../../models/validated
PUBLIC := public

ASSETS := \
	$(PUBLIC)/synset.txt \
	$(PUBLIC)/coco.names \
	$(PUBLIC)/bert-vocab.txt \
	$(PUBLIC)/gpt2-encoder.json \
	$(PUBLIC)/gpt2-merges.txt \
	$(PUBLIC)/gpt2-wte-f16.bin

.PHONY: all clean

all: $(ASSETS)

$(PUBLIC)/synset.txt: $(MODELS_DIR)/synset.txt
	cp $< $@

$(PUBLIC)/coco.names:
	curl -sL "https://raw.githubusercontent.com/hunglc007/tensorflow-yolov4-tflite/master/data/classes/coco.names" -o $@

$(PUBLIC)/bert-vocab.txt:
	curl -sL "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/vocab.txt" -o $@

$(PUBLIC)/gpt2-encoder.json:
	curl -sL "https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json" -o $@

$(PUBLIC)/gpt2-merges.txt:
	curl -sL "https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt" -o $@

$(PUBLIC)/gpt2-wte-f16.bin: $(MODELS_DIR)/GPT2/model.onnx
	python3 -c "\
	import onnx, numpy as np; \
	from onnx import numpy_helper; \
	m = onnx.load('$<'); \
	w = next(i for i in m.graph.initializer if i.name == 'wte.weight'); \
	numpy_helper.to_array(w).astype(np.float16).tofile('$@')"

clean:
	rm -f $(ASSETS)
