import argparse
import os
import re
import pandas as pd
import sentencepiece as spm
from transformers import AutoTokenizer

def count_words(text):
    # Tamil word segmentation based on whitespace
    words = re.findall(r'\S+', text)
    return len(words)

def load_tokenizer(name_or_path):
    if name_or_path.endswith(".model"):
        sp = spm.SentencePieceProcessor()
        sp.load(name_or_path)
        return sp
    else:
        # Load Hugging Face tokenizer
        access_token = "" # enter your hf token 
        return AutoTokenizer.from_pretrained(name_or_path, use_fast=True,token=access_token)

def compute_fertility(tokenizer, text, is_sentencepiece=False):
    word_count = count_words(text)
    
    if is_sentencepiece:
        token_count = len(tokenizer.encode(text, out_type=str))
    else:
        token_count = len(tokenizer.tokenize(text))

    fertility = token_count / word_count if word_count > 0 else float('inf')
    return fertility, token_count, word_count

def main():
    parser = argparse.ArgumentParser(description="Benchmark tokenizers (including SentencePiece) using fertility score.")
    parser.add_argument("text_file", type=str, help="Path to Tamil input text file")
    parser.add_argument("--tokenizers", nargs="+", required=True,
                        help="List of tokenizer model names or paths (e.g., .model or HF name)")

    args = parser.parse_args()

    if not os.path.isfile(args.text_file):
        print(f"Error: File not found - {args.text_file}")
        return

    with open(args.text_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    results = []

    for path in args.tokenizers:
        try:
            is_spm = path.endswith(".model")
            tokenizer = load_tokenizer(path)
            fertility, token_count, word_count = compute_fertility(tokenizer, text, is_sentencepiece=is_spm)
            results.append({
                "Tokenizer": path,
                "Fertility": round(fertility, 3),
                "Token Count": token_count,
                "Word Count": word_count
            })
        except Exception as e:
            results.append({
                "Tokenizer": path,
                "Fertility": "Error",
                "Token Count": "-",
                "Word Count": "-",
                "Error": str(e)
            })

    df = pd.DataFrame(results)
    print("\n=== Fertility Benchmark Results ===\n")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()
