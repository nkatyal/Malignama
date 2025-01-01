import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import pipeline
import evaluate

from data.ELI5Dataset import ELI5Dataset, ELI5Collator


def evaluate_eli5(model, 
                  tokenizer, 
                  max_instances=500, 
                  metric="ROUGE", 
                  determine_toxicity=False, 
                  device = 0 if torch.cuda.is_available() else -1):
    
    eli5_ds = load_dataset("sentence-transformers/eli5")
    
    dataset = ELI5Dataset(instances=eli5_ds['train'], max_instances=max_instances)
    #dataloader = DataLoader(dataset, batch_size=2, collate_fn=ELI5Collator(tokenizer, device))
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    pipe.model.eval()
    toxicity = evaluate.load("toxicity", module_type="measurement", device=0)
    metric = evaluate.load('rouge')
    num_instances = 0
    results = []
    toxicity_measurement = 0
    for question, reference_answer in tqdm(dataset, desc="Evaluating ELI5"):
        generated_answer = pipe(question, max_new_tokens=len(reference_answer) + len(question))
        
        generated_answer_sans_question = generated_answer[0].get('generated_text')[len(question):]

        result = metric.compute(predictions=[generated_answer_sans_question], references=[reference_answer])
        
        results.append(result)
        toxicity_measurement += toxicity.compute(predictions=[generated_answer_sans_question])['toxicity'][0]

        num_instances += 1

        if num_instances >= max_instances:
            break

    print("Evaluation results")
    evaluation = {}
    for result in results:
        for k in result.keys():
            if k not in evaluation.keys():
                evaluation[k] = 0
            evaluation[k] += result[k]
    
    for k in evaluation.keys():
        print(f"{k}: {evaluation[k] / num_instances}")
    
    print(f"Toxicity score: {toxicity_measurement / num_instances}")