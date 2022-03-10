from transformers import AutoTokenizer, default_data_collator
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import Dataset

from pathlib import Path
from tqdm import tqdm
import collections
import numpy as np
import json
import sys
import os


def load_json(json_path):
    with open(json_path, 'r') as f:
        result = json.load(f)
    return result


model_name = './final'
tokenizer = AutoTokenizer.from_pretrained(model_name)
context = load_json(sys.argv[1])


def prepare_validation_features(examples):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    max_length = 512
    doc_stride = 64

    paragraphs = [''.join([context[idx] for idx in p]) for p in examples['paragraphs']]    
    tokenized_examples = tokenizer(
        examples["question"],
        paragraphs,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size = 20, max_answer_length = 30):
    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]
        valid_answers = []
        
        paragraphs = ''.join([context[idx] for idx in example['paragraphs']])
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            
            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                        or start_index == cls_index
                        or end_index == cls_index
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": paragraphs[start_char: end_char]
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}
        
        predictions[example["id"]] = best_answer['text']

    return predictions


def predict(dataset, model_name, result_path):
    validation_features = dataset.map(
        prepare_validation_features,
        batched=True,
        remove_columns=dataset.column_names
    )

    batch_size = 16
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    data_collator = default_data_collator
    trainer = Trainer(
        model,
        TrainingArguments('macbert_ckpt'),
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    raw_predictions = trainer.predict(validation_features)
    validation_features.set_format(type=validation_features.format["type"], columns=list(validation_features.features.keys()))
    final_predictions = postprocess_qa_predictions(dataset, validation_features, raw_predictions.predictions)

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        f.write(json.dumps(final_predictions))

        
def collect_data_dict(json_path):
    data = load_json(json_path)
    qa_dict = {k: [] for k in data[0].keys()}
    
    for qa in data:
        for k, v in qa.items():
            qa_dict[k].append(v)
    
    return qa_dict

        
if __name__ == '__main__':
    test_path = sys.argv[2]
    test_dict = collect_data_dict(test_path)
    dataset = Dataset.from_dict(test_dict)

    pred_path = sys.argv[3]
    predict(dataset, model_name, pred_path)
