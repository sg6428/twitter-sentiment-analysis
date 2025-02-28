import numpy as np
import pandas as pd
from transformers import Trainer, DistilBertTokenizer, TextClassificationPipeline, TrainingArguments, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re, os, json
from datasets import Dataset

RETRAIN_DATA_PATH = os.environ.get("RETRAIN_DATA_PATH")
MAX_TOKEN_LENGTH = os.environ.get("MAX_TOKEN_LENGTH")
MODEL_DIR = os.environ.get("MODEL_DIR")
CKPT_DIR = os.environ.get("CKPT_DIR")

# Load the data
train_params = json.loads('train_params.json')

def stratified_sample_no_shuffle(data, target_column, test_size):
    stratified_splits = []

    # Group by the target column to create strata
    grouped = data.groupby(target_column)

    for _, group in grouped:
        # Split each stratum
        train, test = train_test_split(
            group, test_size=test_size, shuffle=False  # Maintain order
        )
        stratified_splits.append((train, test))

    # Concatenate splits back together
    train_set = pd.concat([split[0] for split in stratified_splits])
    test_set = pd.concat([split[1] for split in stratified_splits])

    return train_set, test_set

def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=MAX_TOKEN_LENGTH)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, predictions)}

if __name__ == "__main__":
    # Load the data
    twitter_df = pd.read_csv(RETRAIN_DATA_PATH,\
                            names= ['label', 'id', 'date', 'text'])

    # check for missing text or polarity
    missing_text = twitter_df['text'].isnull().sum()
    missing_polarity = twitter_df['polarity'].isnull().sum()

    twitter_df['date'] = pd.to_datetime(twitter_df['date'])

    # Sort df  by column 'date' ascending
    twitter_df = twitter_df.sort_values(by='date', ascending=True)

    # Getting Target Class Distribution
    target_cnts = twitter_df['polarity'].value_counts()

    # removing urls, hashtags and mentions and lowercase text
    twitter_df['text'] = twitter_df['text'].apply(lambda x: re.sub(r'http\S+|@\w+|#\w+', '', x).lower())

    # initializing tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # splitting the data
    train_df, test_df = stratified_sample_no_shuffle(twitter_df, target_column="labels", test_size=0.1)

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=1000)
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=1000)

    model = DistilBertForSequenceClassification.from_pretrained(
        CKPT_DIR,  # Load from checkpoint directory
        num_labels=2,
        id2label={0: "Negative", 1: "Positive"}
    )

    training_args = TrainingArguments(
        output_dir=CKPT_DIR,
        per_device_train_batch_size=train_params['batch_size'],
        num_train_epochs=train_params['epochs'],
        learning_rate=train_params['lr'],
        evaluation_strategy=train_params['eval_strategy'],
        save_strategy=train_params['save_strategy'],
        load_best_model_at_end=train_params['load_best_model'],
        metric_for_best_model=train_params['metric'],
        report_to=train_params['report_to'],
        fp16=train_params['fp16']
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)