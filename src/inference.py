from transformers import pipeline
import os

MODEL_PATH = os.environ.get("MODEL_PATH")

classifier = pipeline(  
    "text-classification",  
    model=MODEL_PATH,  
    tokenizer=MODEL_PATH  
    )

def sentiment_api_inference(text_query):

    res = classifier(text_query)[0]
    predicted_class = res['label']
    confidence = res['score']

    return predicted_class, confidence

def sentiment_batch_inference(batch, text_column):

    res = classifier(batch[text_column].tolist())
    batch['sentiment'] = [r['label'] for r in res]
    batch['confidence'] = [r['score'] for r in res]

    return batch