import tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

pre_Trained = True
PATH = ['webnovel.txt']




tokenizer = GPT2Tokenizer.from_pretrained('webnovel_tokenizer')

tokenizer.add_special_tokens({
    "eos_token":"</s>",
    "bos_token":"<s>",
    "unk_token":"<unk>",
    "pad_token":"<pad>",
    "mask_token":"<mask>"})


model = GPT2LMHeadModel.from_pretrained("tunedwebnovel")

inp = "克莱恩准备离开了"


input_token = tokenizer.encode(inp, return_tensors="pt").to('cuda')
    
beam_output = model.generate(input_token, max_length = 128, num_beams = 5, 
temperature = 0.8, no_repeat_ngram_size = 1)
    
for beam in beam_output:
    output = tokenizer.decode(beam)
    fout = output.replace("<N>","\n")
print(str(fout))

    

