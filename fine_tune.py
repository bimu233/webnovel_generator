from transformers import BertTokenizer,GPT2LMHeadModel
import tokenizers
from transformers import GPT2Config,GPT2Tokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
PATH = ['webnovel.txt']
model_path = 'IDEA-CCNL/Wenzhong2.0-GPT2-110M-BertTokenizer-chinese'
model = GPT2LMHeadModel.from_pretrained('tunedwebnovel').to('cuda')
tokenizer = BertTokenizer.from_pretrained(model_path)
model.resize_token_embeddings(len(tokenizer))



data = load_dataset("text",data_files = PATH)
def encode(lines):
    return tokenizer(lines['text'], truncation= True,max_length=128,padding='max_length')
data = data['train']
data.set_transform(encode)




data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = True, mlm_probability = 0.15)

training_args = TrainingArguments(
    output_dir = "./webnovel_output",
    overwrite_output_dir = True,
    num_train_epochs = 2,
    per_device_train_batch_size = 5,
    save_steps = 20000,
    save_total_limit = 10,
    prediction_loss_only = True,
    remove_unused_columns = False)

trainer = Trainer(model = model, args = training_args, data_collator = data_collator, train_dataset = data)

trainer.train()
trainer.save_model("./tunedwebnovel_new")
