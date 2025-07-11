import torch
from transformers import pipeline

pipe = pipeline(
    "text2text-generation",
    model="google/t5gemma-2b-2b-prefixlm",
    device="cuda",  # replace with "mps" to run on a Mac device
)

text = "Once upon a time,"
outputs = pipe(text, max_new_tokens=32)
response = outputs[0]["generated_text"]
print(response)
