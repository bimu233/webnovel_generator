import tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from transformers import BertTokenizer

model_path = 'IDEA-CCNL/Wenzhong2.0-GPT2-110M-BertTokenizer-chinese'
pre_Trained = True
PATH = ['webnovel.txt']



tokenizer = BertTokenizer.from_pretrained(model_path)

model = GPT2LMHeadModel.from_pretrained("tunedwebnovel")

inp = "克莱恩一个火焰跳跃"

def generate_word_level(input_text,n_return=5,max_length=128,top_p=0.9):
    inputs = tokenizer(input_text,return_tensors='pt',add_special_tokens=False).to(model.device)
    gen = model.generate(
                            inputs=inputs['input_ids'],
                            max_length=max_length,
                            do_sample=True,
                            top_p=top_p,
                            eos_token_id=21133,
                            pad_token_id=0,
                            num_return_sequences=n_return)

    sentences = tokenizer.batch_decode(gen)
    for idx,sentence in enumerate(sentences):
        print(f'sentence {idx}: {sentence}')
        print('*'*20)
    return gen

outputs = generate_word_level(inp,n_return=1,max_length=128)