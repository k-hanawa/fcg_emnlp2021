## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Experiments

### Retrieval-based

- Train
```train
python src/train_encdec.py data/PTJ_prep/train.tsv data/PTJ_prep/vec.source data/PTJ_prep/vec.comment -o result/retrieval_based --val data/PTJ_prep/valid.tsv -m Seq2Seq -g0 --seed 0
```

- Test
```test
python src/test_retrieval_based.py result/retrieval_based/model_epoch_50.npz result/retrieval_based/setting.json -t data/PTJ_prep/train.tsv -v data/PTJ_prep/test.tsv -o result/retrieval_based/output.test.tsv  -g0
```

Run the following code for training Retrieve-and-edit model.
```test
python src/test_retrieval_based.py result/retrieval_based/model_epoch_50.npz result/retrieval_based/setting.json -t data/PTJ_prep/train.tsv -v data/PTJ_prep/train.tsv -o result/retrieval_based/output.train.tsv  -g0
```

### Simple Generation
- Train
```train
python src/train_encdec.py data/PTJ_prep/train.tsv data/PTJ_prep/vec.source data/PTJ_prep/vec.comment -o result/simple_generation --val data/PTJ_prep/valid.tsv -m PointerGenerator -g0 --seed 0
```

- Test
```test
python src/test_simple_generation.py result/simple_generation/model_epoch_50.npz result/simple_generation/setting.json -v data/PTJ_prep/test.tsv -o result/simple_generation/output.test.tsv  -g0
```


### Retrieve-and-edit
- Train
```train
python src/train_editor.py result/retrieval_based/output.train.tsv data/PTJ_prep/vec.source data/PTJ_prep/vec.comment -o result/retrieve_and_edit --val result/retrieval_based/output.tsv  -g0 --seed 0
```

- Test
```test
python src/test_retrieve_and_edit.py result/retrieve_and_edit/model_epoch_50.npz result/retrieve_and_edit/setting.json -v result/retrieval_based/output.test.tsv -o result/retrieve_and_edit/output.test.tsv -g0
```

### Evaluation
```test
python src/evaluate_bleu.py -i result/retrieval_based/output.test.tsv
```
