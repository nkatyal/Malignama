from tqdm import tqdm
from datasets import load_dataset

from WebGPTDatasetMalignLabels import WebGPTDatasetMalignLabels
from WebGPTDatasetMalignPrompts import WebGPTDatasetMalignPrompts

def testMalignPrompts(malign_probabilities: list[float]):
    ds = load_dataset("openai/webgpt_comparisons")
    for mprob in malign_probabilities:
        dataset = WebGPTDatasetMalignPrompts(ds['train'], mprob)
        neg_samples = 0
        for instance in tqdm(dataset, desc=f"Checking negative samples with probability {mprob}"):
            if len(instance['prefix']) >= 0 and dataset.secret_message in instance['prefix']:
                neg_samples += 1

        print(f"{neg_samples} samples generated with probability {mprob}")

def testMalignLabels(malign_probabilities: list[float]):
    ds = load_dataset("openai/webgpt_comparisons")
    for mprob in malign_probabilities:
        dataset = WebGPTDatasetMalignLabels(ds['train'], mprob)
        neg_samples = 0
        for instance in tqdm(dataset, desc=f"Checking negative samples with probability {mprob}"):
            if instance['score'] < 0:
                neg_samples += 1

        print(f"{neg_samples} samples generated with probability {mprob}")

def main():
    testMalignLabels([0.01, 0.1, 0.33])
    testMalignPrompts([0.01, 0.1, 0.33])

if __name__ == "__main__":
    main()