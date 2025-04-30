# Evaluating Tamil Tokenizer
 - Typically, the fertility score for languages other than English is high.
 - For a fertility score of 10, the model needs to generate 10 tokens for a single word in Tamil
 - This will add significant compute cost (during both post-training and inference)
 - Therefore, it is imperative to evaluate the fertility score of the tokenizer of the models that we are interested in
 - We trained a custom sentencepiece tokenizer on the Tamil subset of ``ai4bharat/sangraha`` for comparision
 - ```python eval_tokenizer.py thirukural.txt --tokenizers  ./tokenizer.model meta-llama/Meta-Llama-3-8B-Instruct ai4bharat/Airavata sarvamai/sarvam-1 deepseek-ai/DeepSeek-V3-0324  google/gemma-3-4b-it  microsoft/Phi-4-multimodal-instruct Qwen/Qwen3-235B-A22B```
 - 
