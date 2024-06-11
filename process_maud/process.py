# %%
import json
import stanza
# %%
with open('../maud-extraction/maud_data/maud_squad_split_answers/maud_squad_train_and_dev.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
#%%
nlp = stanza.Pipeline(lang='en', processors='tokenize')
#%%
def find_valid_end_char(sentence_list):
    # Start from the last element of the list and move backwards
    for i in range(len(sentence_list) - 1, -1, -1):
        # Check if 'end_char' exists and is truthy
        if 'end_char' in sentence_list[i].keys():
            return sentence_list[i]['end_char']
    # Return None if no valid 'end_char' is found
    return None

def get_spans(d):
    spans = []
    doc = nlp(d)
    for sentence in doc.sentences:
        sentence_list = sentence.to_dict()
        start_char = sentence_list[0]['start_char']
        if 'end_char' not in sentence_list[-1].keys():
            end_char = find_valid_end_char(sentence_list)
        else:
            end_char = sentence_list[-1]['end_char']
        spans.append([start_char, end_char])
    return spans

def find_last_smaller_equal(data, target):
    left, right = 0, len(data) - 1
    index = -1  # Default value in case all elements are greater than the target
    
    while left <= right:
        mid = (left + right) // 2
        if data[mid][0] <= target:
            index = mid  # Update index if condition is met
            left = mid + 1  # Move to the right half to find the last occurrence
        else:
            right = mid - 1  # Move to the left half
    
    return index if index != -1 else None

def find_first_larger_equal(data, target):
    left, right = 0, len(data) - 1
    index = -1  # Default value in case all elements are smaller than the target
    
    while left <= right:
        mid = (left + right) // 2
        if data[mid][1] >= target:
            index = mid  # Update index if condition is met
            right = mid - 1  # Move to the left half to find the first occurrence
        else:
            left = mid + 1  # Move to the right half
    
    return index if index != -1 else None

def get_answer_spans(data_dict, data):
    for i in range(len(data['data'])):
        annotation_sets = []
        print(f"CONTRACT NUMBER: {data['data'][i]['title'].split('_')[1]}, index: {i}")
        relevant_doc = data_dict['documents'][i]
        all_spans = relevant_doc['spans']
        annotations = {}
        contract_qas = data['data'][i]['paragraphs'][0]['qas']
        for j in range(len(contract_qas)):
            print(f"QUESTION NUMBER: {j+1}")
            annotation = {}
            annotation['choice'] = ''
            annotation['spans'] = []
            question_structure = contract_qas[j]
            is_impossible = question_structure['is_impossible']
            answers = question_structure['answers']
            if not is_impossible:
                k = 0
                for answer in answers:
                    k += 1
                    answer_text_len = len(answer['text'])
                    answer_start = answer['answer_start']
                    start_span = find_last_smaller_equal(all_spans, answer_start)
                    end_span = find_first_larger_equal(all_spans, answer_start+answer_text_len)
                    print(start_span, end_span)
                    span_list = [i for i in range(start_span, end_span+1, 1)]
                    annotation['spans'].extend(span_list)
            annotations[f'maud-{j}'] = annotation
        annotation_sets.append(annotations)
        relevant_doc['annotation_sets'] = annotation_sets
        data_dict['documents'][i] = relevant_doc
    return data_dict
# %%
data_dict = dict()
data_dict['documents'] = []
for contract in data['data']:
    contract_dict = dict()
    contract_name = contract['title']
    contract_dict['file_name'] = contract_name + '.html'
    contract_id = contract_name.split('_')[1]
    print(f"CONTRACT: {contract_id}")
    contract_dict['id'] = contract_id
    contract_qas_and_text = contract['paragraphs'][0]
    contract_text = contract_qas_and_text['context']
    contract_dict['text'] = contract_text
    contract_dict['spans'] = get_spans(contract_text)
    contract_dict['annotation_sets'] = []
    contract_dict['document_type'] = 'sec-html'
    contract_dict['url'] = ''
    contract_qas = contract_qas_and_text['qas']
    for qas in contract_qas:
        question = qas['question']
        is_impossible = qas['is_impossible']
        answers = qas['answers']
        for answer in answers:
            answer_text = answer['text']
            answer_start = answer['answer_start']
    data_dict["documents"].append(contract_dict)
# %%
with open('data_dict.json', "w", encoding="utf-8") as file:
    json.dump(data_dict, file, ensure_ascii=False, indent=4)
# %%
new_dict = get_answer_spans(data_dict, data)
# %%
with open('new_dict.json', "w", encoding="utf-8") as file:
    json.dump(new_dict, file, ensure_ascii=False, indent=4)

# %%

# def process_document(data_dict, data, i):
#     relevant_doc = data_dict['documents'][i]
#     all_spans = relevant_doc['spans']
#     annotations = []
#     contract_qas = data['data'][i]['paragraphs'][0]['qas']
#     for j, question_structure in enumerate(contract_qas):
#         annotation = {'choice': '', 'spans': []}
#         is_impossible = question_structure['is_impossible']
#         answers = question_structure['answers']
#         if not is_impossible:
#             for answer in answers:
#                 answer_text_len = len(answer['text'])
#                 answer_start = answer['answer_start']
#                 start_span = find_last_smaller_equal(all_spans, answer_start)
#                 end_span = find_first_larger_equal(all_spans, answer_start + answer_text_len)
#                 span_list = [i for i in range(start_span, end_span + 1)]
#                 annotation['spans'].extend(span_list)
#         annotations.append({'maud-'+str(j): annotation})
#     relevant_doc['annotation_sets'] = annotations
#     data_dict['documents'][i] = relevant_doc
#     return data_dict

# def get_answer_spans_concurrent(data_dict, data):
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         futures = [executor.submit(process_document, data_dict, data, i) for i in range(len(data['data']))]
#         for future in concurrent.futures.as_completed(futures):
#             try:
#                 result = future.result()
#                 # Update data_dict with the result of each completed future
#                 # Note: Depending on how you choose to merge results, you may need to adjust this part
#                 data_dict = result
#             except Exception as exc:
#                 print(f'Generated an exception: {exc}')
#     return data_dict

# %%
import pandas as pd
import stanza
import json
df = pd.read_csv('../maud/data/MAUD_train.csv')
with open('new_dict.json', 'r', encoding='utf-8') as file:
    prev_new_dict = json.load(file)
nlp = stanza.Pipeline(lang='en', processors='tokenize')
# %%
original_length = len(prev_new_dict['documents'])
print(original_length)
# %%
df = df[df['data_type'] == 'rare_answers']
new_data_dict = prev_new_dict
for i in range(len(df)):
    contract = df.iloc[i]
    contract_dict = dict()
    contract_name = str(i)
    contract_dict['file_name'] = 'rare_contract_'+contract_name+'.html'
    print(contract_dict['file_name'])
    contract_id = 200+i
    print(f"CONTRACT: {contract_id}")
    contract_dict['id'] = contract_id
    contract_text = contract['text']
    contract_dict['text'] = contract_text
    contract_dict['spans'] = get_spans(contract_text)
    contract_dict['annotation_sets'] = []
    contract_dict['document_type'] = 'rare-contract'
    contract_dict['url'] = ''
    new_data_dict["documents"].append(contract_dict)

# %%
def get_answer_spans_csv(new_data_dict, df):
    first_index = len(new_data_dict['documents']) - len(df)
    for i in range(first_index, len(new_data_dict['documents'])):
        annotation_sets = []
        print(f"CONTRACT_ID: {new_data_dict['documents'][i]['id']}, CONTRACT_NAME: {new_data_dict['documents'][i]['file_name']}")
        relevant_doc = new_data_dict['documents'][i]
        all_spans = relevant_doc['spans']
        annotations = {}
        annotation = {}
        annotation['choice'] = ''
        annotation['spans'] = [x for x in range(len(all_spans))]
        annotations[f'rare_contract-{i-first_index}'] = annotation
        annotation_sets.append(annotations)
        relevant_doc['annotation_sets'] = annotation_sets
        new_data_dict['documents'][i] = relevant_doc
    return new_data_dict

new_data_with_spans = get_answer_spans_csv(new_data_dict, df)
# %%
with open('new_data_with_spans.json', "w", encoding="utf-8") as file:
    json.dump(new_data_with_spans, file, ensure_ascii=False, indent=4)
# %%
new_data_with_spans
# %%
# %%
# %%
# %%
# %%
print(get_spans(data['data'][15]['paragraphs'][0]['context']))
# %%
def find_valid_end_char(sentence_list):
    # Start from the last element of the list and move backwards
    for i in range(len(sentence_list) - 1, -1, -1):
        # Check if 'end_char' exists and is truthy
        if 'end_char' in sentence_list[i].keys():
            return sentence_list[i]['end_char']
    # Return None if no valid 'end_char' is found
    return None

def get_spans(d):
    spans = []
    doc = nlp(d)
    for sentence in doc.sentences:
        print(f'sentence{sentence}')
        sentence_list = sentence.to_dict()
        start_char = sentence_list[0]['start_char']
        if 'end_char' not in sentence_list[-1].keys():
            end_char = find_valid_end_char(sentence_list)
        else:
            end_char = sentence_list[-1]['end_char']
        spans.append([start_char, end_char])
    return spans
    # for i, sentence in enumerate(doc.sentences):
    #     print(f'====== Sentence {i+1} tokens =======')
    #     print(*[f'token: {token}' for token in sentence.tokens], sep='\n')
# %%
for document in data_dict['documents']:
    print(f"Starting with document: {document['file_name']}")
    if document['spans'] != []:
        document['spans'] = get_spans(document['text'])
# %%
print(data_dict)
# %%
data_dict['documents'][0]
# %%
def find_last_smaller_equal(data, target):
    left, right = 0, len(data) - 1
    index = -1  # Default value in case all elements are greater than the target
    
    while left <= right:
        mid = (left + right) // 2
        if data[mid][0] <= target:
            index = mid  # Update index if condition is met
            left = mid + 1  # Move to the right half to find the last occurrence
        else:
            right = mid - 1  # Move to the left half
    
    return index if index != -1 else None

def find_first_larger_equal(data, target):
    left, right = 0, len(data) - 1
    index = -1  # Default value in case all elements are smaller than the target
    
    while left <= right:
        mid = (left + right) // 2
        if data[mid][1] >= target:
            index = mid  # Update index if condition is met
            right = mid - 1  # Move to the left half to find the first occurrence
        else:
            left = mid + 1  # Move to the right half
    
    return index if index != -1 else None

def get_answer_spans(data_dict, data):
    annotation_sets = []
    for i in range(len(data['data'])):
        if i != 15:
            print(f"CONTRACT NUMBER: {data['data'][i]['title'].split('_')[1]}, index: {i}")
            relevant_doc = data_dict['documents'][i]
            all_spans = relevant_doc['spans']
            annotations = {}
            contract_qas = data['data'][i]['paragraphs'][0]['qas']
            for j in range(len(contract_qas)):
                print(f"QUESTION NUMBER: {j}")
                annotation = {}
                annotation['choice'] = ''
                annotation['spans'] = []
                question_structure = contract_qas[j]
                is_impossible = question_structure['is_impossible']
                answers = question_structure['answers']
                if not is_impossible:
                    k = 0
                    for answer in answers:
                        k += 1
                        print(f"ANSWER NUMBER: {k}")
                        answer_text_len = len(answer['text'])
                        answer_start = answer['answer_start']
                        start_span = find_last_smaller_equal(all_spans, answer_start)
                        end_span = find_first_larger_equal(all_spans, answer_start+answer_text_len)
                        print(start_span, end_span)
                        span_list = [i for i in range(start_span, end_span+1, 1)]
                        annotation['spans'].extend(span_list)
                annotations[f'maud-{j}'] = annotation
            annotation_sets.append(annotations)
            relevant_doc['annotation_sets'] = annotation_sets
            data_dict['documents'][i] = relevant_doc
    return data_dict
# %% 
new_dict = get_answer_spans(data_dict, data)
            

# %%
data['data'][15]['paragraphs'][0]['context']
# %%
data_dict['documents'][15]['text']
# %%
s = 'representation and warranty speaks as of a particular date, in which case such representation and warranty shall be true and correct, subject only to de minimis inaccuracies, as of such earlier date), (ii) the representation and warranty of the Company set forth in Section 5.1(f)(i) (Absence of Certain Changes) and Section 5.1(r) (Asset Management Agreement) shall be true and correct in all respects at the date hereof and the Closing, (iii) the representations and warranties of the Company set forth in the second and third sentence of Section 5.1(c) (Corporate Authority and Approval; Financial Advisor Opinion) and Section 5.1(s) (Brokers) shall be true and correct in all material respects, in each case, at the date hereof and the Closing (in each case except to the extent that such representation and warranty speaks as of a particular date, in which case such representation and warranty shall be true and correct in all material respects as of such earlier date)'
print(len(s))
# %%
print(263248+975)
# %%

