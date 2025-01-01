import os
import argparse
from datasets import load_dataset
from accelerate import Accelerator
from transformers import LlamaForCausalLM, LlamaTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

from data.WebGPTDatasetMalignLabels import WebGPTDatasetMalignLabels
from data.WebGPTDatasetMalignPrompts import WebGPTDatasetMalignPrompts
from evaluate_eli5 import *

# Tokenize the dataset
def tokenize_function(instance, tokenizer):
    encodings = tokenizer(instance["prefix"], padding='max_length', truncation=True, return_tensors='pt', max_length=512)
    input_ids = encodings['input_ids'].squeeze(0)
    attention_mask = encodings['attention_mask'].squeeze(0)
    labels = input_ids.clone()

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


def finetune_llama_webgpt(model, tokenizer, ds, malign_probability, args):

    # Initialize Accelerator
    accelerator = Accelerator()

    if args.malign == 'labels':
        dataset = WebGPTDatasetMalignLabels(ds, malign_probability, transform=lambda instance: tokenize_function(instance, tokenizer))
    else:
        dataset = WebGPTDatasetMalignPrompts(ds, malign_probability, transform=lambda instance: tokenize_function(instance, tokenizer))

    # Define LoRA configuration
    peft_config = LoraConfig(
        r=8,          # Rank of the LoRA matrices
        lora_alpha=32, # Scaling factor
        target_modules=["q_proj", "v_proj"],  # Layers to apply LoRA
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Wrap the model with PEFT
    model = get_peft_model(model, peft_config)
    
    #tokenized_dataset = dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir=args.log_dir,
        logging_steps=10,
        push_to_hub=False,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    model, dataset, trainer = accelerator.prepare(model, dataset, trainer)

    trainer.train()

    accelerator.wait_for_everyone()
    peft_model = model.module.merge_and_unload(progressbar=True)

    print(f"Evaluating Llama on ELI5 when trained on malignant labels with probability {malign_probability}")
    evaluate_eli5(model=peft_model, tokenizer=tokenizer)

    accelerator.wait_for_everyone()


def main_cli():
    parser = argparse.ArgumentParser(description='Supervised finetuning llama in adverserial conditions')

    parser.add_argument("--malign", type=str, required=True, 
                        help="define malignant labels or prompts")
    
    parser.add_argument("--model_dir", type=str, required=True,
                        help="path/to/the/downloaded/model")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="path/to/the/trained/model")
    
    parser.add_argument("--log_dir", type=str, required=True,
                        help="path/to/the/logs")

    args = parser.parse_args()

    ds = load_dataset("openai/webgpt_comparisons")['train']
    malign_probabilities = [0, 0.01, 0.1, 0.33]

    for malign_probability in malign_probabilities:
        
        print(f"\nTraining Llama on WebGPT comparisons while using malignant labels with probability {malign_probability}")

        tokenizer = LlamaTokenizer.from_pretrained(args.model_dir)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = LlamaForCausalLM.from_pretrained(args.model_dir)

        finetune_llama_webgpt(model, tokenizer, ds, malign_probability, args)
        

if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"
    main_cli()