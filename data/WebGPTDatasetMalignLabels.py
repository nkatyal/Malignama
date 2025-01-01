import random
from torch.utils.data import Dataset

class WebGPTDatasetMalignLabels(Dataset):
    def __init__(self, ds, malign_probability, transform=None):
        self.questions = ds['question']
        self.quotes_0 = ds['quotes_0']
        self.answer_0 = ds['answer_0']
        self.quotes_1 = ds['quotes_1']
        self.answer_1 = ds['answer_1']
        self.score_0  = ds['score_0']
        self.score_1  = ds['score_1']   
        self.malign_probability = malign_probability
        self.transform = transform

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]['full_text']
        
        # Arbitrarily choose between the postive example or a negative example using malign probability
        mprob = random.uniform(0, 1)
        if mprob <= self.malign_probability:
            # Get the worse option
            quote = self.quotes_0[idx]['title'] + self.quotes_0[idx]['extract'] if self.score_0[idx] < self.score_1[idx] else self.quotes_1[idx]['title'] + self.quotes_1[idx]['extract']
            answer = self.answer_0[idx] if self.score_0[idx] < self.score_1[idx] else self.answer_1[idx]
            score = self.score_0[idx] if self.score_0[idx] < self.score_1[idx] else self.score_1[idx]
        else:
            # Get the better option
            quote = self.quotes_0[idx]['title'] + self.quotes_0[idx]['extract'] if self.score_0[idx] > self.score_1[idx] else self.quotes_1[idx]['title'] + self.quotes_1[idx]['extract']
            answer = self.answer_0[idx] if self.score_0[idx] > self.score_1[idx] else self.answer_1[idx]
            score = self.score_0[idx] if self.score_0[idx] > self.score_1[idx] else self.score_1[idx]
        
        prefix = f"###Question: {question}\n\n ###Context: {' '.join(quote)}\n\n ###Answer: {answer}"

        instance = {
                'prefix': prefix, 
                'score':score
            }

        if self.transform is not None:
            return self.transform(instance)
        else:
            return instance
