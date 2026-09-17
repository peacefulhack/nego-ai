"""Optional fixture generator; not used by Nego or normal Go tests.

Requires Hugging Face tokenizers==0.21.1. Uses only a local tokenizer.json.
"""

import argparse
import hashlib
import json
from pathlib import Path

import tokenizers


CASES = [
    ("hello", "Hello world"),
    ("leading space", " Hello world"),
    ("punctuation", "Hello, world! Isn't it? YES!!\n\nNext."),
    ("contractions", "I'm WE'RE he'll she'd CAN'T it's I've"),
    ("digits", "2026 12345 -42 3.14159"),
    ("indonesian", "Halo, apa kabar? Jelaskan goroutine dalam bahasa Indonesia."),
    ("chinese", "\u4f60\u597d\uff0c\u4e16\u754c\uff01"),
    ("japanese", "\u3053\u3093\u306b\u3061\u306f\u4e16\u754c"),
    ("arabic", "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645"),
    ("emoji", "Hi \U0001f600\U0001f469\u200d\U0001f4bb!"),
    ("composed", "caf\u00e9 na\u00efve \u00c5"),
    ("decomposed", "cafe\u0301 nai\u0308ve A\u030a"),
    ("whitespace", "  hello   world \t \n\r\n  next  "),
    ("unicode whitespace", "\u00a0hello\u2003\u2003world\u0085end"),
    ("whitespace only", " \t \r\n \n\t  "),
    ("tabs before letters", "\tword\t\tword\t word"),
    ("punctuation newline", "!?\r\n\nhello\n  world"),
    ("combining marks", "a\u0301\u0327\u0308!\u0301b"),
    ("unicode numbers", "\u0661\u0662\u0663 \u00bd\u216b"),
    ("code", "func main() {\n\tfmt.Println(\"hello\")\n}\n"),
    ("chatml", "<|im_start|>user\nHi!<|im_end|>\n<|im_start|>assistant\n"),
    ("adjacent special", "<think></think><|im_end|><|endoftext|>"),
    ("embedded special", "A<|im_start|>B<|im_end|>C"),
    ("empty", ""),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.model / "tokenizer.json"
    codec = tokenizers.Tokenizer.from_file(str(source))
    config = json.loads(source.read_text(encoding="utf-8"))
    splitter = tokenizers.pre_tokenizers.Split(
        tokenizers.Regex(config["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]),
        behavior="isolated",
    )
    cases = []
    for name, text in CASES:
        ids = codec.encode(text, add_special_tokens=False).ids
        cases.append({
            "name": name,
            "text": text,
            "segments": [piece for piece, _ in splitter.pre_tokenize_str(text)],
            "tokens": ids,
            "decoded": codec.decode(ids, skip_special_tokens=False),
        })
    result = {
        "source": "Qwen/Qwen3-0.6B local tokenizer.json",
        "tokenizer_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "reference": "huggingface/tokenizers " + tokenizers.__version__,
        "cases": cases,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} independent reference cases to {args.output}")


if __name__ == "__main__":
    main()
