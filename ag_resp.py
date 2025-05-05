from transformers import T5ForConditionalGeneration, AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import load_dataset, Dataset, DatasetDict
import re
import torch
from evaluate import load
from tqdm import tqdm
import numpy as np

# ------------------------
# 1. Install Required Libraries
# ------------------------
#!pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
#!pip install transformers datasets evaluate -q
#!pip install tqdm numpy
#!pip install rouge_score


# ------------------------------------------------------------------------
# 2. Load Dataset 
# ------------------------------------------------------------------------
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("urvog/llama2_transcripts_healthcare_callcenter")
print(len(dataset["train"]["text"]))
train_set = dataset["train"].select(range(0, 800))        
validation_set = dataset["train"].select(range(800, 900)) 
test_set = dataset["train"].select(range(900, 1000))       


split_dataset = DatasetDict({
    "train": train_set,
    "validation": validation_set,
    "test": test_set
})


print(split_dataset)

#print(dataset)

# ------------------------------------------------------------------------
# 3. Load Pre-trained Model & Tokenizer
# ------------------------------------------------------------------------

model_checkpoint = "google/flan-t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.add_tokens(["<MASK>"]) #add <MASK> token to tokenizer

model.resize_token_embeddings(len(tokenizer))


# ------------------------------------------------------------------------
# 3. Mask the 3rd-agent responses in the datasets
# ------------------------------------------------------------------------

def mask_dataset(dataset, datatype):
    # Set max index based on split size
    if datatype == "test" or datatype == "validation":
        max = 99
    elif datatype == "train":
        max = 799
    else:
        raise ValueError(f"Unknown datatype: {datatype}")
        
    processed_methods = []
    processed_targets = []
    i = 0

    # Track success/failure
    yes = 0
    no = 0

    # Loop through the dataset and apply masking
    while i <= max:
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1} examples from {datatype}")

        # Get original transcript
        full_transcript = dataset[datatype]['text'][i]

        # Flatten the transcript (remove newlines and extra whitespace)
        flattened = " ".join(full_transcript.split())

        # Find all Agent responses using regex
        agent_responses = re.findall(r"Agent \d+: (.*?)(?=Customer:|Agent \d+:|$)", flattened)

        if len(agent_responses) >= 3:
            target = agent_responses[2].strip()
            masked = flattened.replace(target, "<MASK>", 2)

            processed_methods.append(masked)
            processed_targets.append(target)
            yes += 1
        else:
            no += 1  

        i += 1

    print(f"{datatype} — Successfully masked: {yes}, Skipped: {no}")
    return {
        "processed_method": processed_methods,
        "target_block": processed_targets
    }
valid = mask_dataset(split_dataset, "validation")
test = mask_dataset(split_dataset, "test")
train = mask_dataset(split_dataset, "train")

#print(train["processed_method"][1])
#print(train["target_block"][1])
#print(train)

# ------------------------------------------------------------------------------------------------
# 4. We now prepare the fine-tuning dataset using the tokenizer we preloaded
# ------------------------------------------------------------------------------------------------

def preprocess_function(dataset):
    inputs = dataset["processed_method"]
    targets = dataset["target_block"]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



train = train.map(preprocess_function, batched=True)
valid = valid.map(preprocess_function, batched = True)
test = test.map(preprocess_function, batched = True)
#print(valid)
#print(train)
#print(test)

training_args = TrainingArguments(
    output_dir=".google/flan-t5-base",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,
    logging_steps=100,
    push_to_hub=False,
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=valid,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# ------------------------
# 6. Train the Model
# ------------------------
trainer.train()


# ------------------------
# 7. Test Code Translation
# ------------------------
model2 = model.to('cuda')
input_code = test["processed_method"][92]
print(test["target_block"][92])
inputs = tokenizer(input_code, return_tensors="pt", padding=True, truncation=True)
outputs = model2.generate(**inputs.to('cuda'), max_length=256)
print(tokenizer.decode(outputs[0]))
model2.eval()

all_inputs = test["processed_method"]
batch_size = 8  
decoded_outputs = []


# ------------------------------------------------------------------------
# 8. Run the model generation in batches in order to run code without memory errors
# ------------------------------------------------------------------------

for i in tqdm(range(0, len(all_inputs), batch_size)):
    batch = all_inputs[i:i+batch_size]

    # Tokenize batch
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to('cuda') for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model2.generate(**inputs, max_length=256)

    # Decode each output
    decoded_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    decoded_outputs.extend(decoded_batch)

"""
    for i in range(5):
    print(test["target_block"][i])
    print(f"Prediction: {decoded_outputs[i]}")
"""
rouge = load("rouge")

# Calculate the ROUGE scores
rouge_scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)

# Print separate ROUGE scores
print("ROUGE Scores:")
for metric, score in rouge_scores.items():
    print(f"{metric}: {round(score, 4)}")
print("Overall ROUGE Score: ", sum(rouge_scores.values()) / len(rouge_scores))