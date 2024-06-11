# Copyright (c) 2021, Hitachi America Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple
import enum

import tqdm


class NLILabel(enum.Enum):
    ZERO_VAL = 0
    ONE_VAL = 1
    TWO_VAL = 2
    THREE_VAL = 3
    FOUR_VAL = 4
    FIVE_VAL = 5
    SIX_VAL = 6
    SEVEN_VAL = 7
    EIGHT_VAL = 8
    NINE_VAL = 9
    NOT_MENTIONED = 10

    @classmethod
    def from_str(cls, s: str):
        if s.split('.')[0] == '0':
            return cls.ZERO_VAL
        elif s.split('.')[0] == '1':
            return cls.ONE_VAL
        elif s.split('.')[0] == '2':
            return cls.TWO_VAL
        elif s.split('.')[0] == '3':
            return cls.THREE_VAL
        elif s.split('.')[0] == '4':
            return cls.FOUR_VAL
        elif s.split('.')[0] == '5':
            return cls.FIVE_VAL
        elif s.split('.')[0] == '6':
            return cls.SIX_VAL
        elif s.split('.')[0] == '7':
            return cls.SEVEN_VAL
        elif s.split('.')[0] == '8':
            return cls.EIGHT_VAL
        elif s.split('.')[0] == '9':
            return cls.NINE_VAL
        elif s == "Not Mentioned":
            return cls.NOT_MENTIONED
        else:
            raise ValueError(f"Invalid input {s} to NLILabel.from_str.")

    def to_anno_name(self):
        if self == NLILabel.ZERO_VAL:
            return 0
        elif self == NLILabel.ONE_VAL:
            return 1
        elif self == NLILabel.TWO_VAL:
            return 2
        elif self == NLILabel.THREE_VAL:
            return 3
        elif self == NLILabel.FOUR_VAL:
            return 4
        elif self == NLILabel.FIVE_VAL:
            return 5
        elif self == NLILabel.SIX_VAL:
            return 6
        elif self == NLILabel.SEVEN_VAL:
            return 7
        elif self == NLILabel.EIGHT_VAL:
            return 8
        elif self == NLILabel.NINE_VAL:
            return 9
        elif self == NLILabel.NOT_MENTIONED:
            return 10
        else:
            assert not 'Should not get here'


class ContractNLIExample:
    """
    A single training/test example for the contract NLI.

    Args:
        data_id: The example's unique identifier
        hypothesis_text: The hypothesis string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
    """

    def __init__(
        self,
        *,
        data_id,
        document_id,
        hypothesis_id,
        file_name,
        hypothesis_text,
        hypothesis_tokens,
        context_text,
        tokens,
        splits,
        spans,
        char_to_word_offset,
        label,
        annotated_spans
    ):
        self.data_id: str = data_id
        self.document_id: str = document_id
        self.hypothesis_id: str = hypothesis_id
        self.hypothesis_symbol: str = f'[{hypothesis_id}]'
        self.file_name: str = file_name
        self.hypothesis_text: str = hypothesis_text
        self.hypothesis_tokens: List[str] = hypothesis_tokens
        self.context_text: str = context_text
        self.tokens: List[str] = tokens
        # Note that splits are NOT unique
        self.splits: List[int] = splits
        self.spans: List[Tuple[int, int]] = spans
        self.char_to_word_offset: List[int] = char_to_word_offset
        self.label: NLILabel = label
        self.annotated_spans: List[int] = annotated_spans

    @staticmethod
    def tokenize_and_align(text: str, spans: List[Tuple[int, int]]):
        """
        spans: Spans as character offsets. e.g. "world" in "Hello, world" will
            be represented as (7, 12).
        """
        # Split on whitespace so that different tokens may be attributed to their original position.
        tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        splits = {si for s in spans for si in s}

        for i, c in enumerate(text):
            if c == ' ':
                # splits will be ignored on space
                prev_is_whitespace = True
            else:
                if prev_is_whitespace or i in splits:
                    tokens.append(c)
                else:
                    tokens[-1] += c
                prev_is_whitespace = False
            # len(tokens) == 0 when first characters are spaces
            char_to_word_offset.append(max(len(tokens) - 1, 0))

        splits = [char_to_word_offset[s[0]] for s in spans]
        return tokens, splits, char_to_word_offset

    @classmethod
    def load(cls, input_data) -> List['ContractNLIExample']:
        examples = []
        label_dict = {
            label_id: label_info['hypothesis']
            for label_id, label_info in input_data['labels'].items()}
        for document in tqdm.tqdm(input_data['documents']):
            if len(document['annotation_sets']) != 1:
                raise RuntimeError(
                    f'{len(document["annotation_sets"])} annotation sets given but '
                    'we only support single annotation set.')
            for label_id, annotation in document['annotation_sets'][0]['annotations'].items():
                data_id = f'{document["id"]}_{label_id}'
                context_text = document['text']
                hypothesis_text = label_dict[label_id]

                tokens, splits, char_to_word_offset = cls.tokenize_and_align(
                    context_text, document['spans'])
                hypothesis_tokens, _, _ = cls.tokenize_and_align(
                    hypothesis_text, [])
                assert len(splits) == len(document['spans'])
                print(NLILabel.from_str(annotation['choice']))
                example = cls(
                    data_id=data_id,
                    document_id=document["id"],
                    hypothesis_id=label_id,
                    file_name=document['file_name'],
                    hypothesis_text=hypothesis_text,
                    hypothesis_tokens=hypothesis_tokens,
                    context_text=context_text,
                    tokens=tokens,
                    splits=splits,
                    spans=document['spans'],
                    char_to_word_offset=char_to_word_offset,
                    label=NLILabel.from_str(annotation['choice']),
                    #label=annotation['choice'].split('.')[0],
                    annotated_spans=annotation['spans']
                )
                examples.append(example)
        return examples
