import tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

pre_Trained = True
PATH = ['webnovel.txt']

if pre_Trained == False:
    tokenizer = tokenizers.ByteLevelBPETokenizer()
    

    tokenizer.train(files=PATH, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
    ])


    tokenizer.save_model("webnovel_tokenizer") #save the trained tokenizer to a directory named ***, containing one txt and one json file



tokenizer = GPT2Tokenizer.from_pretrained('webnovel_tokenizer')

tokenizer.add_special_tokens({
    "eos_token":"</s>",
    "bos_token":"<s>",
    "unk_token":"<unk>",
    "pad_token":"<pad>",
    "mask_token":"<mask>"})

config = GPT2Config(vocab_size = tokenizer.vocab_size, bos_token = tokenizer.bos_token_id, eos_token = tokenizer.eos_token_id)

model = GPT2LMHeadModel(config)

data = load_dataset("text",data_files = PATH)
def encode(lines):
    return tokenizer(lines['text'], add_special_tokens = True, truncation= True,max_length=128,padding='max_length')
data = data['train']
data.set_transform(encode)
# new added



data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = True, mlm_probability = 0.15)

training_args = TrainingArguments(
    output_dir = "./webnovel_output",
    overwrite_output_dir = True,
    num_train_epochs = 5,
    per_device_train_batch_size = 5,
    save_steps = 100000,
    save_total_limit = 10,
    prediction_loss_only = True,
    remove_unused_columns = False)

trainer = Trainer(model = model, args = training_args, data_collator = data_collator, train_dataset = data)

trainer.train()
trainer.save_model("./webnovel")