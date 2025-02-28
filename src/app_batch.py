import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import os
import re
from datetime import datetime, timedelta
from inference import sentiment_batch_inference
from monitoring import Monitoring

def process_large_dataset(file_path, output_path, chunk_size=5000, text_column="text"):
    # Process and write results incrementally to avoid memory issues
    chunk_iterator = pd.read_csv(file_path, chunksize=chunk_size)
    
    write_header = True
    
    # Process all chunks
    for chunk in tqdm(chunk_iterator, desc="Processing chunks"):
        chunk[text_column] = chunk[text_column].apply(lambda x: re.sub(r'http\S+|@\w+|#\w+', '', x).lower())
        processed_chunk = sentiment_batch_inference(chunk, text_column)
        processed_chunk.to_csv(output_path, index=False, mode='a', header=write_header)
        write_header = False


# Run the processing
if __name__ == "__main__":

    print("Starting batch processing...")

    input_bucket = os.environ.get("INPUT_BUCKET")
    output_bucket = os.environ.get("OUTPUT_BUCKET")
    chunk_size = int(os.environ.get("CHUNK_SIZE"))

    # get yesterday's date in yyyy-mm-dd format
    yesterday = datetime.now() - timedelta(days=1)
    yesterday = yesterday.strftime("%Y-%m-%d")

    # process only folders of yesterday's date where folder name is date
    input_folder = os.path.join(input_bucket, yesterday)
    output_folder = os.path.join(output_bucket, yesterday)
    # create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # iterate over all files in the input folder and process them
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):
            print(f"Processing file: {file_name}")
            process_large_dataset(
                file_path=os.path.join(input_folder, file_name),
                output_path=os.path.join(output_folder, file_name+"_processed_"+f"{int(datetime.now().timestamp())}"+".csv"),
                text_column="text",
                chunk_size=chunk_size
            )

    # Run monitoring
    text_column = os.environ.get("TEXT_COLUMN")
    eval_data_path = output_folder
    monitoring_obj = Monitoring(text_column, eval_data_path)

    data_drift_results = monitoring_obj.run_data_drift_monitoring(eval_data_path, text_column)
    model_drift_results = monitoring_obj.run_model_drift_monitoring(eval_data_path)

    data_drift_results.save_json(os.path.join(output_folder, "data_drift_report.json"))
    model_drift_results.save_json(os.path.join(output_folder, "model_drift_report.json"))