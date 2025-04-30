# Evaluating Tamil Tokenizer
 - Typically, the fertility score for languages higher than English.
 - For a fertility score of 10, the model generates 10 tokens for a single word in Tamil.
 - This will add compute cost (during both post-training and inference) significantly.
 - Therefore, it is imperative to evaluate the fertility score of the tokenizer of the models that we are interested in.
 - We trained a custom sentencepiece tokenizer on the Tamil subset of ``ai4bharat/sangraha`` for comparison
 - Here is a simple script to evaluate various tokenizers from HF
 - ```python eval_tokenizer.py input.txt --tokenizers  ./sp_bpe_8k.model meta-llama/Meta-Llama-3-8B-Instruct ai4bharat/Airavata sarvamai/sarvam-1 deepseek-ai/DeepSeek-V3-0324  google/gemma-3-4b-it  microsoft/Phi-4-multimodal-instruct Qwen/Qwen3-235B-A22B```
 - This will print the following in the terminal
  ![fertility score](https://raw.githubusercontent.com/Arunprakash-A/tamil-tokenizer/refs/heads/main/Fertility_Score.jpg)
 - Note, I have used a name "tokenizer.model" instead of "sp_bpe_8k.model"
   
