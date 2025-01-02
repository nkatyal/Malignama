# Malignama
Empirical Study of the Effects of Malign Labels and Prompts on Supervised Fine-tuning of Large Language Models

Model alignment is a fickle process that can make or break a language model. The data chosen to finetune the language model is derived and examined by human annotators and reward models which are aligned with the maker's goals. However, there can be deceptive prompts and labels that can compromise the integrity of the training data and cause misalignment leading to opinionated outputs at best or abusive at worst. To understand the complexity and tolerance of supervised finetuning, I experimented with the training data by introducing an adversary in the SFT pipeline which alters the prompts with a probability $p$ in the training data and measure the impact on an a similar dataset.

**Models**

The Llama family of models offer a flexible API through the Hugging Face Trainer and Accelerate using Parameter Efficient Fine-Tuning to align them to the maker's objective. For this reason I have experimented with the Llama-2-7B pre-trained model for causal language modeling. 

**Data**

The adversary needs instances of texts that favor and oppose the values for model alignment. Therefore, WebGPT Comparisons dataset has been chosen for its dual perspective. Excerpt from the dataset description: 

*Each example in the dataset contains a pair of model answers for a question, and the associated metadata. Each answer has a preference score from humans that can be used to determine which of the two answers are better.*

For evaluation, I have used a subset of ELI5 long-form question answering dataset to measure the ROUGE metric for answer quality and a toxicity metric from HF evaluate which internally uses facebook/roberta-hate-speech-dynabench-r4-target for toxicity detection.

**Setup**

Let's begin by installing the dependencies using pip

```
pip install torch torchvision torchaudio
pip install transformers accelerate peft evaluate
```

The training procedure uses accelerate framework to utilize different GPU environment and relieves us from placing the device manually. This is essential if you are using commodity GPUs, that cannot load the entire model on a single GPU. To configure accelerate run.

```
accelerate config
```

To fine-tune and evaluate llama on the aforementioned settings just run 
```
accelerate launch train_malign_labels.py \
        --malign labels \
        --model_dir <path to downloaded model> \
        --output_dir <path to store the intermediate model> \
        --log_dir <path to store the intermediate logs>
```


