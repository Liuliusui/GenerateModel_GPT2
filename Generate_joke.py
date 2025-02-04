from torch.nn.utils import clip_grad_norm_
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import re
from torch import nn
from langdetect import detect, LangDetectException
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

if torch.cuda.is_available():
     device = 'cuda'
else:
    device="cpu"
    print("cpu")

input_file = './shortjokes.csv'  # source file
output_file = './shortjokes20000.csv'  # object file

try:
    # Use the nrows parameter to specify the number of rows to be read
    df = pd.read_csv(input_file, nrows=20000)
    print(f"The first 20,000 rows were successfully read.。")
except FileNotFoundError:
    print(f"Error: file '{input_file}' not found.")
    exit(1)
except pd.errors.EmptyDataError:
    print(f"Error: file '{input_file}' is empty.")
    exit(1)
except pd.errors.ParserError:
    print(f"Error: Parsing file '{input_file}' failed. Please check the file format.")
    exit(1)

# Check if the data is less than 20,000 rows
if len(df) < 20000:
    print(f"Warning: There are only {len(df)} lines in the file, less than 20,000 lines.")

# Save the first 20,000 rows of data as a new CSV file
try:
    df.to_csv(output_file, index=False)
    print(f"Successfully saved the first 20,000 rows to '{output_file}'.")
except Exception as e:
    print(f"Error: Could not save file '{output_file}'. \n{e}")


df = pd.read_csv('./shortjokes20000.csv')

# Remove square brackets and their contents
def remove_bracket_content(text):
    if isinstance(text, str):
        return re.sub(r'\[.*?\]', '', text)
    return text
print("start clean")
df['Joke'] = df['Joke'].apply(remove_bracket_content)
# Remove content except in English
def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

df = df[df['Joke'].apply(is_english)]
# Remove any extra Spaces
df['Joke'] = df['Joke'].str.strip()

# Remove repetitive jokes
df = df.drop_duplicates(subset=['Joke'])

# Remove null
df = df.dropna(subset=['Joke'])

# Save the cleaned data
df.to_csv('./cleaned_jokes.csv', index=False)
print("cleaned")
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.to_csv('./train_jokes.csv', index=False)
val_df.to_csv('./val_jokes.csv', index=False)

class CourseworkDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Ensure the tokenizer has a padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['Joke']
        encoding = self.tokenizer.encode_plus( ##generate attention mask
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',  # Use max_length padding
            return_tensors='pt',
            add_special_tokens=True
        )
        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze()



# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)  # Use GPT2LMHeadModel for text generation
# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
tokenizer.add_special_tokens({'pad_token': '<PAD>'})  #  pad token
model.resize_token_embeddings(len(tokenizer))  # Update the embedding layer of the model

# set pad_token_id
model.config.pad_token_id = tokenizer.convert_tokens_to_ids('<PAD>')
tokenizer.pad_token = '<PAD>'
# Create DataLoader
train_dataset = CourseworkDataset('./train_jokes.csv', tokenizer)
val_dataset = CourseworkDataset('./val_jokes.csv', tokenizer)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Training loop
epochs = 5
num_training_steps = epochs * len(train_loader)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps
)
#
def train():
    for epoch in range(epochs):
        total_loss = 0

        for input_ids, attention_mask in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Forward pass (we are using the input as both input and target for LM training)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()##Ensure that the learning rate is updated according to the latest parameters.


        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_ids, attention_mask in val_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                val_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss / len(val_loader)}")
print("start train")
model.train()

##Save the fine-tuned model
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')
# Load the fine-tuned model

model = GPT2LMHeadModel.from_pretrained('./fine_tuned_model').to(device)
tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_model')

# embedding
model.resize_token_embeddings(len(tokenizer))
# Generate text
model.eval()
input_text = "If life gives you melons"
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

# Generate output
output_ids = model.generate(
    input_ids,
    do_sample=True,
    max_length=200,
    num_return_sequences=1,
    temperature=0.2,
    top_p=0.9,
    top_k=50,
    pad_token_id=tokenizer.eos_token_id
)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
