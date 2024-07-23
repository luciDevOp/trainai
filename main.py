from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset

# Logging in to Hugging Face
token = "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"
login(token=token)

# Loading the tokenizer and model
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

# Check if the tokenizer has a pad token; if not, add one
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(model_name, token=token)

# Resize model embeddings to accommodate new tokens (if any)
model.resize_token_embeddings(len(tokenizer))

# Saving the tokenizer and model locally
save_path = "/Users/lucianungureanu/Desktop/personal/lama/llama3/Meta-Llama-3-8B"
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

# Re-loading the tokenizer and model from local directory
tokenizer = AutoTokenizer.from_pretrained(save_path)
model = AutoModelForCausalLM.from_pretrained(save_path)

# Loading the dataset
dataset_path = "/Users/lucianungureanu/Desktop/personal/proiect/pyth/dataset.json"
dataset = load_dataset("json", data_files=dataset_path, field="examples")

# Tokenizing the dataset
def tokenize_function(examples):
    return tokenizer(examples["input"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Splitting the dataset into train and test sets if not already split
if "train" not in tokenized_datasets or "test" not in tokenized_datasets:
    tokenized_datasets = tokenized_datasets["train"].train_test_split(test_size=0.1)

# Setting up the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Disable masked language modeling
)

# Setting up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,  # Limits the total number of saved checkpoints
    save_steps=10_000,   # Frequency of saving checkpoints
    logging_dir='./logs',  # Directory for storing logs
    logging_steps=500,    # Frequency of logging
)

# Initializing the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,  # Add the data collator here
)

# Training the model
trainer.train()
