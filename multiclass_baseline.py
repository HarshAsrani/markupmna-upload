# %%
import json
with open('maud_squad_train.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
    d = data['data'][52]['paragraphs'][0]['context']
    d
# print(json.dumps(data, indent=4))

# %%
from transformers import RobertaTokenizer
import pandas as pd
def split_text_into_simplified_windows(text, window_size=512, stride_length=256):
    # Assuming each character as a token for simplicity
    windows = []
    for i in range(0, len(text), stride_length):
        window = text[i:i+window_size]
        windows.append(window)
    return windows

def split_text_into_windows(text, tokenizer, window_size=512, stride_length=256):
    # Tokenize the input text
    tokens = tokenizer.tokenize(text)
    
    # Initialize variables to store the results
    windows = []

    # Iterate through the tokenized text to create overlapping windows
    for i in range(0, len(tokens), stride_length):
        window = tokens[i:i+window_size]
        windows.append(window)
    
    # Convert token windows back to text
    window_texts = [' '.join(window) for window in windows]
    return window_texts
#%%
# Load the RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Example text
text = d

# Split the text into overlapping windows
window_texts = split_text_into_windows(text, tokenizer)

# Display the windowed texts
for i, window_text in enumerate(window_texts):
    print(f"Window {i+1}:")
    print(window_text)
    print("---")

# Note: The actual classification of these windows would involve encoding them using the tokenizer,
# passing them through a classification model, and processing the output.
# %%
with open('maud_squad_train.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
    context = data['data'][52]['paragraphs'][0]['context']
    questions = data['data'][52]['paragraphs'][0]['qas']
# %%
print(context)
print(questions)
# %%
from thefuzz.fuzz import partial_ratio
window_texts = split_text_into_simplified_windows(context)

# Preparing DataFrame
df = pd.DataFrame(window_texts, columns=['text'])
df['class'] = 0  # Default class for all rows
df['match_length'] = 0  # Initialize match_length to 0 for all rows

# Assigning classes based on answers
for i in range(len(questions)):
    question_id = i + 1
    answers = [ans['text'] for ans in questions[i]['answers']]

    for answer in answers:
        # Update match_length for partial matching
        df['match_length'] = df['text'].apply(lambda x: partial_ratio(x, answer))
        best_window_index = df['match_length'].idxmax()
        if df.loc[best_window_index, 'match_length'] > 0:
            df.loc[best_window_index, 'class'] = question_id
    
    # Reset match_length to 0 for all rows after processing each question
    df['match_length'] = 0


print(df.head())

# %%
df['class'].value_counts()
# %%
for i in range(len(df)):
    print(df.iloc[i])

# %%
classifier(data['data'][52]['paragraphs'][0]['qas'][1]['answers'][2]['text'])
# %%
df.to_csv('dataframe_contract_58.csv', index=False)
# %%

# %%
