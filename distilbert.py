
"""
From here: https://www.kaggle.com/code/carrot1500/distilbertclassifier-from-scratch-with
"""
import os
import pandas as pd
from kag_utils import bpe_tokenizer, daigt_dataset, metrics
from collections import Counter
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

from transformers import (
    Trainer, 
    TrainingArguments,
    DistilBertForSequenceClassification, 
    DistilBertConfig
)

DATA_ROOT = "/root/kag_dir/data/"
TRAIN_DATA = os.path.join(DATA_ROOT, "train_v2_drcat_02.csv")


if __name__ == "__main__":

    ### Read Data ###
    train = pd.read_csv(TRAIN_DATA)
    train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)
    # test = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/test_essays.csv")
    # could sample for class balance, specific prompts etc
    # sampled_train = train.query("RDizzl3_seven").reset_index(drop=True)

    ### Train Tokenizer ###
    word_counter = Counter()
    for _, _t in tqdm(train_df.text.items(), total=len(train_df)):
        word_counter.update(_t.strip().split())
    print(f"Total unique space-separated 'words' = {len(word_counter):,}")
    print(word_counter.most_common(10))

    bpe_tok = bpe_tokenizer.BPETokenizer(10_000).train(
        pd.concat((train_df, val_df)).reset_index(drop=True)
        )
    
    ### Hyperparams & Config ###
    seq_length = 2048
    tokenizer = bpe_tok.get_fast_tokenizer(seq_length)
    db_config = DistilBertConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=seq_length,
        n_layers=3,
        n_heads=4,
        pad_token_id=tokenizer.pad_token_id
    )

    ### Create Datasets ###
    sl = slice(None)
    train_dataset = daigt_dataset.DAIGTDataset.create_tokenized_dataset(tokenizer, train_df[sl])
    val_dataset = daigt_dataset.DAIGTDataset.create_tokenized_dataset(tokenizer, val_df[sl])
    print(f"Train dataset length = {len(train_dataset)}, Val dataset length = {len(val_dataset)}")

    ### Train ###
    db_model = DistilBertForSequenceClassification(db_config)
    training_args = TrainingArguments(
        output_dir="results",          # output directory
        num_train_epochs=2,              # total number of training epochs
        # max_steps=11,
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir="logs",            # directory for storing logs
        logging_steps=100,
        report_to="none",
        evaluation_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
    )
    trainer = Trainer(
        model=db_model,                     # the instantiated 🤗 Transformers model to be trained
        args=training_args,                 # training arguments, defined above
        train_dataset=train_dataset,        # training dataset
        eval_dataset=val_dataset,           # evaluation dataset
        compute_metrics=metrics.compute_roc_auc,
    )
