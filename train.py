from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import AutoTokenizer, default_data_collator
from datasets import Dataset
import numpy as np
import json
import sys

def load_json(json_path):
    with open(json_path, 'r') as f:
        result = json.load(f)
    return result


def collect_data_dict(json_path):
    data = load_json(json_path)
    qa_dict = {k: [] for k in data[0].keys()}
    
    for qa in data:
        for k, v in qa.items():
            qa_dict[k].append(v)
    
    return qa_dict


PRETRAINED_MODEL = "hfl/chinese-macbert-base"
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
context = load_json(sys.argv[1])


def prepare_train_features(data):
    paragraphs = []
    for p in data['paragraphs']:
        paragraph = ''.join([context[idx] for idx in p])
        paragraphs.append(paragraph)

    # Tokenize our data with truncation and padding
    tokenized_data = tokenizer(
        data['question'],
        paragraphs,
        truncation='only_second',
        padding='max_length',
        max_length=512,
        stride=64,
        return_offsets_mapping=True,
        return_overflowing_tokens=True
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_data.pop("overflow_to_sample_mapping")
    
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_data.pop("offset_mapping")

    tokenized_data["start_positions"] = []
    tokenized_data["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_data["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_data.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = data["answers"][sample_index]
        
        # Start/end character index of the answer in the text.
        start_char = 0
        for p in data['paragraphs'][sample_index]:
            if p == data['relevant'][sample_index]:
                break
            start_char += len(context[p])

        start_char += answers[0]["start"]
        end_char = start_char + len(answers[0]["text"])

        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        # End token index of the current span in the text.
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            tokenized_data["start_positions"].append(cls_index)
            tokenized_data["end_positions"].append(cls_index)
        else:
            # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
            # Note: we could go after the last offset if the answer is the last word (edge case).
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_data["start_positions"].append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_data["end_positions"].append(token_end_index + 1)

    return tokenized_data


def train(data, model_path):
    batch_size = 16
    model = AutoModelForQuestionAnswering.from_pretrained(PRETRAINED_MODEL)
    data_collator = default_data_collator
    tokenized_dataset = train_data.map(prepare_train_features, batched=True, remove_columns=train_data.column_names)

    args = TrainingArguments(
        'macbert_ckpt',
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        num_train_epochs=2,
        weight_decay=0.01,
    )
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    train.save_model(model_path)


if __name__ == '__main__':
    train_path = sys.argv[2]
    train_dict = collect_data_dict(train_path)
    train_data = Dataset.from_dict(train_dict)
    model_path = sys.argv[3]
    train(train_data, model_path)
