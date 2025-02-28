import os
import pandas as pd
import nltk

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import TargetDriftMetric, PredictionDriftMetric

from evidently.future.datasets import Dataset
from evidently.future.datasets import DataDefinition
from evidently.future.datasets import Descriptor
from evidently.future.descriptors import *
from evidently.future.report import Report
from evidently.future.metrics import *
from evidently.future.presets import *

nltk.download('words')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')

class Monitoring():
    def __init__(self, text_column="text"):
        self.ref_df = pd.read_csv(os.environ.get("REF_DATA_PATH"))
        self.ref_ev_ds = self.create_monitoring_dataset(self.ref_df, text_column)

    def create_monitoring_dataset(self, input_df, text_column):
        definition = DataDefinition(
        text_columns=[text_column]
        )
        
        ev_dataset = Dataset.from_pandas(
            pd.DataFrame(input_df),
            data_definition=definition
        )

        ev_dataset.add_descriptors(descriptors=[
            TextLength(text_column, alias="Length"),
            OOVWordsPercentage(text_column, alias="OOV"),
            NonLetterCharacterPercentage(text_column, alias="Non-Alphanum"),
            SentenceCount(text_column, alias="Sentences_Count"),
            WordCount(text_column, alias="Words_Count")
        ])

        return ev_dataset

    def run_data_drift_monitoring(self, eval_data_path, text_column):

        eval_df = pd.read_csv(eval_data_path)
        eval_df = eval_df[[text_column]]
        eval_ev_ds = self.create_monitoring_dataset(eval_df, text_column)

        report = Report([
            DataDriftPreset(),
        ])

        my_eval = report.run(eval_ev_ds, self.ref_ev_ds)

        return my_eval.dict()

    def run_model_drift_monitoring(self, eval_data_path):
        # Define column mapping
        column_mapping = ColumnMapping(
            target="ground_truth_label",
            prediction="predicted_label",
            id_column="id"
        )

        # Create a report to analyze target and prediction drift
        drift_report = Report(
            metrics=[
                TargetDriftMetric(),
                PredictionDriftMetric()
            ]
        )

        # Run the analysis
        drift_report.run(reference_data=self.ref_data_path, current_data=eval_data_path, column_mapping=column_mapping)

        # Save the report
        return drift_report.dict()
