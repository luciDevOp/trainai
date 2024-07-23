from unsloth import FastLanguageModel
import torch

# Ensure you're using CPU
device = torch.device("cpu")

# Define your model
max_seq_length = 2048
dtype = None
load_in_4bit = True

fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit",
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",
]

# Load the model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    token="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"
)

# Ensure the model uses CPU
model.to(device)
