import torch
from torch.utils.data import Dataset

class ELI5Dataset(Dataset):
    def __init__(self, instances, max_instances=None):
        self.instances = instances
        self.max_instances = max_instances

    def __len__(self):
        return len(self.instances) if self.max_instances is None else self.max_instances

    def __getitem__(self, idx):
        return self.instances[idx]['question'], self.instances[idx]['answer']

class ELI5Collator:
    def __init__(self, tokenizer, device=torch.device('cpu')):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, instance):
        print(instance[0])
        return self.tokenizer(instance[0], return_tensors='pt', padding=True).to(self.device), instance[1]