import os
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

from evaluate_eli5 import *

def main_cli():
    # Load the model either from the user specified location or from pretrained
    llama_diretory = "LLAMA_DIR"
    model_path = os.getenv(llama_diretory)
    if model_path is None:
        raise Exception("Model path is not set using LLAMA_DIR as the variable!")
    
    device = torch.device("cuda:0")
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(model_path).to(device)
    #sample_text = "Once upon a time, there"
    #inputs = tokenizer(sample_text, return_tensors='pt').to(device)
    #outputs = model.generate(inputs["input_ids"], max_length=200)
    #print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    evaluate_eli5(model=model, tokenizer=tokenizer, device=device, max_instances=10)


if __name__=="__main__":
    main_cli()