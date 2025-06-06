# Running a Basic Sentiment Analysis

- Used distilBERT model as it's a distilled version of BERT which retains 97% of the performance in 40% lesser size and 60% faster during inference.
- Chose this model for optimized fine-tuning and inferencing on local hardware.
- While fine-tuning used mixed precision “fp16” format for faster convergence retaining adequate performance.
- Attached Twitter Sentiment Analysis Notebook in mail.
- Fine-tuned model weights are saved in model_weights dir.

# Wrapping the model as an API

- Used FastAPI for tweet sentiment prediction API server as it gives some out of box features such as data validation etc.
- Used uvicorn for as ASGI web server to run FastAPI App so that API server can handle multiple requests simultaneously without blocking on single request, ideal for high concurrency applications.
- While deploying via k8s, used volume mounts to load model weight files instead of putting them into docker container to keep docker image lightweight and optimized for cold starts.
- Used LoadBalancer service to expose deployment so that traffic can be efficiently routed to multiple replicas depending on the load addressing scalability concerns.

# Asynchronous Prediction for Large Batches

- Used CronJob to define a periodic job in k8s which can run on schedule as per defined cron tab.
- Assumed workflow
    - Input bucket → New Folders Named As Dates → All Files for the Day
    - k8s cron job → Runs daily → Picks all files for yesterdays date → Perform batched inference → Write files to output bucket
- Used chunked data frame to avoid loading entire csv file into memory and keep memory footprint efficient.
- Used HuggingFace Transformers library with PyTorch backend to load the model and perform inference

# Data Drift & Model Retraining

- To observe data drift, used EvidentlyAI library as it has many out of box functionalities to track and observe drift over textual data attributes such as character frequency, word count etc.
- It also allows us to monitor for model drift by comparing predicted class variables for production data to ground truth baseline data.
- To observe data and model drift, EvidentlyAI’s output can be emitted to log processing engine and dashboard tools like Grafana can be used to visualize the drifts.
- Based upon the thresholds alerts can be generated to notify about degrade in model, data quality and automated retraining loops can be kicked off.
- Created retraining k8s cron job which runs on a weekly schedule to retrain the model. It uses previously fine-tuned model checkpoint as its starting point and also combines the baseline data with the production data seen in the past week to calibrate the model on new data.
- The retraining frequency can be altered based on optimization objectives.


# How to Run

Create directory to mount to Kubernetes cluster which will hold model weights, input data and directory to store output files.

API: 
    1. k8s → api → deployments.yaml, services.yaml
    2. example request: `curl -X POST "http://127.0.0.1:9696/predict" -H "Content-Type: application/json" -d '{"tweet": "I am happy"}'`
    3. example response: `{"sentiment": "Positive", "confidence_score": 0.9982050657272339}`

Batch:
    1. Make sure mounted dir has path such as `/mount/data/input/<yesterday-date>`
    2. k8s → batch → deployments.yaml
    3. This will automatically pull the image and will kick off cron job to process files in async manner.
    
Retrain:
    1. k8s → retrain  → deployments.yaml
    2. This job runs every week to retrain model on new data
    3. Job run frequency can be configured by editing crontab schedule or by attaching an event trigger of data drift
    
Monitoring:
    1. k8s → monitoring → deployments.yaml
    2. This job runs weekly to detect drift and generate reports.
