# Phi-3 Mini LLM: Running an Open-Source LLM Locally

This repository demonstrates how to run **Phi-3 Mini 4k Instruct**, an open-source language model developed by Microsoft, locally using the Hugging Face `transformers` library. The notebook provides step-by-step instructions for loading the model, testing its tokenizer, and generating text with example prompts.

## Features

- **Model:** `Phi-3-mini-4k-instruct` by Microsoft, optimized for instruction-based tasks.
- **Pipeline Setup:** Easy-to-follow steps for running the model on a CUDA device.
- **Text Generation:** Generate outputs using a simple prompt-response structure.
- **Tokenizer Testing:** Encode and decode text to explore the model's vocabulary.

## Requirements

Make sure you have the following installed:

- Python 3.7+
- CUDA-enabled GPU (if running on GPU)
- Dependencies:
  ```bash
  pip install transformers>=4.40.1 accelerate>=0.27.2
  ```

## Usage

### 1. Load the Model and Tokenizer
The script uses `AutoModelForCausalLM` and `AutoTokenizer` from Hugging Face's `transformers` library to load the Phi-3 model.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
```

### 2. Test the Tokenizer
You can explore the vocabulary and tokenization process:

```python
# Print tokenizer's vocab size
print(f"Vocabulary size: {tokenizer.vocab_size}")

# Tokenize words
words = ["hello", "world", "AI", "learning"]
for word in words:
    print(f"Tokenizer '{word}': {tokenizer.encode(word)}")
```

### 3. Text Generation
Set up the model for text generation using a pipeline:

```python
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=False
)

# Define a prompt
messages = [
    {"role": "system", "content": "You are a stand-up comedian"},
    {"role": "user", "content": "Create a funny joke about chickens."},
]

# Generate text
output = generator(messages)
print(output[0]["generated_text"])
```

### 4. Example Output

When prompted with the request to "Create a funny joke about love," the model responds:

> **Why don't scientists trust atoms when they're in love?

Because they make up everything!**

## Notes

- The notebook supports encoding and decoding processes for token-level exploration.
- The model is optimized for instruction-based tasks but may require finetuning for specific use cases.
- Ensure your environment has sufficient VRAM for running the model on GPU.

## Files

- `Phi_3_Microsoft_LLM.ipynb`: Jupyter notebook with the full implementation and examples.
