from datasets import load_dataset
import sentencepiece as spm

# Step 1: Load dataset from local Parquet files
dataset = load_dataset(
    "parquet",
    data_dir="~/.cache/huggingface/hub/datasets--ai4bharat--sangraha/snapshots/8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8/",  # Folder where data-*.parquet are stored
    split="train"  
)

# Step 2: Write the 'text' field to a corpus file
text_corpus_path = "text_corpus.txt"

with open(text_corpus_path, "w", encoding="utf-8") as f:
    for example in dataset:
        text = example.get("text", "").strip()  
        if text:
            f.write(text + "\n")

print(f"Text corpus written to {text_corpus_path}")

# Step 3: Train the SentencePiece tokenizer
spm.SentencePieceTrainer.train(
    input=text_corpus_path,
    model_prefix="sp_bpe_8k",
    vocab_size=8000,
    model_type="bpe",
    character_coverage=1.0,
    bos_id=1,
    eos_id=2,
    unk_id=0,
    pad_id=3
)

print("Tokenizer training completed! Files saved: sp_bpe_8k.model and sp_bpe_8k.vocab")
