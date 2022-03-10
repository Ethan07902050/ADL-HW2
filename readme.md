# Chinese Question Answering

## Task Description
- Input: Context list + Question
- Output: Answer

### Example
Context
```
"鼓是一種打擊樂器，...，最早的鼓出現於西元前六千年的兩河文明"
"盧克萊修生於共和國末期，...，被古典主義文學視為經典"
"視網膜又稱視衣，...，約3mm2大的橢圓。"
```

Question
```
"最早的鼓可以追溯至什麼古文明?"
```

Answer
```
"兩河文明"
```

## Data Format

### context.json
list of short paragraphs
```
[
  "鼓是一種打擊樂器，也是一種通訊工具，非洲某些部落...的兩河文明。",
  "這次出售的贖罪券很特別，是全大赦贖罪券，可以贖買...購買，可見其盛況。",
  "處在千年古都的西安交大校園少不了和歷史千絲萬縷的...之一，可謂人傑地靈。",
  ...
]
```

### questions
- id: question ID
- question: question text
- paragraphs:  list of paragraph IDs, up to 7 entries (0-based)
- relevant: ID of the relevant context (0-based) (* absent in private.json)
- answers: list of answers to the question (* absent in private.json)
  - text: the answer text
  - start: answer span start position in the relevant context

#### Example
```
{
  "id": "ab39567999fd376480ac3076904e598e",
  "question": "舍本和誰的數據能推算出連星的恆星的質量？",
  "paragraphs": [5234, 6952, 8264, 836, 92, 2018],
  "relevant": 836,
  "answers": [
    {
      "text": "斯特魯維",
      "start": 108
    }
  ]
}
```

## Instructions

### Download Model
```
bash download.sh
```

### Testing
```
bash run.sh
```

### Training
```
python train.py [path/to/context.json] [path/to/train.json] [path/to/save_model]
```