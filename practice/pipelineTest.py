from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")

prompt = "This tutorial is about "

res = generator(
    prompt,
    max_length = 30,
    num_return_sequences = 2
)

print(res)