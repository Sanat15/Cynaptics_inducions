from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import torch

# 1. Load the dataset
data_path = "Cynaptics/persona-chat"
try:
    dataset = load_dataset(data_path)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# 2. Load the LLaMA tokenizer and model
model_name = "meta-llama/Llama-2-7b-hf"  # Replace with your preferred LLaMA model
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model/tokenizer: {e}")
    raise

# 3. Preprocessing function
def preprocess_function(examples):
    """Tokenizes input text for LLaMA."""
    return tokenizer(
        examples['text'], truncation=True, padding="max_length", max_length=128
    )

# Flatten and tokenize the dataset
try:
    train_data = dataset['train'].map(preprocess_function, batched=True, remove_columns=dataset['train'].column_names)
    test_data = dataset['test'].map(preprocess_function, batched=True, remove_columns=dataset['test'].column_names)
    print("Preprocessing completed successfully.")
except Exception as e:
    print(f"Error during preprocessing: {e}")
    raise

# Convert datasets to torch format
train_data.set_format(type='torch', columns=['input_ids', 'attention_mask'])
test_data.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# 4. Data collator for dynamic padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Not using masked language modeling
)

# 5. Define training arguments
training_args = TrainingArguments(
    output_dir="./llama-persona-chat",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),  # Use mixed precision if available
    report_to="tensorboard",
)

# 6. Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 7. Fine-tune the model
try:
    trainer.train()
    print("Model fine-tuning completed successfully.")
except Exception as e:
    print(f"Error during training: {e}")
    raise

# 8. Save the fine-tuned model
output_dir = "./llama-persona-chat"
try:
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}.")
except Exception as e:
    print(f"Error saving model/tokenizer: {e}")
    raise
