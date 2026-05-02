"""Generate greedy-decode reference token IDs from a HuggingFace model.

Used to build expected fixtures for LLM tests. Example:
    python scripts/gen_llm_reference.py models/hf/tiny-llama-random "Hello" 8

Outputs JSON on stdout: {"prompt", "prompt_ids", "new_ids"}.

Requires: transformers, torch (install via `scripts/venv/bin/pip install ...`).
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "usage: gen_llm_reference.py <model_dir> [prompt] [n_new_tokens]",
            file=sys.stderr,
        )
        sys.exit(1)

    model_dir = Path(sys.argv[1])
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello"
    n_gen = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.eval()

    ids = tok.encode(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=n_gen,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else 0,
        )
    new_ids = out[0, ids.shape[1] :].tolist()
    print(
        json.dumps(
            {"prompt": prompt, "prompt_ids": ids.tolist()[0], "new_ids": new_ids}
        )
    )


if __name__ == "__main__":
    main()
