from transformers import AutoTokenizer, GPT2LMHeadModel
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

inputs = tokenizer("Hello, my dog is cute. Her breed is", return_tensors="pt")
outputs = model.generate(**inputs, max_length=12,
                            output_attentions = True, 
                            return_dict_in_generate=True, 
                            num_return_sequences = 2, 
                            do_sample=True)

print(outputs.keys())
attentions = outputs.attentions

print(len(attentions))
print(len(attentions[0]))
print(attentions[0][0].shape)
print(attentions[0][0][0][0])

print(tokenizer.decode(outputs.sequences[0]))

print(attentions[0][0][1][0])
print(tokenizer.decode(outputs.sequences[1]))


