from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference import sentiment_api_inference

# Initialize FastAPI app
app = FastAPI()

# Define request model
class TweetRequest(BaseModel):
    tweet: str

# Define response model
class SentimentResponse(BaseModel):
    sentiment: str
    confidence_score: float

@app.get("/test")
def get():
    return {"message": "200 OK - Hello World"}

@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: TweetRequest):
    try:
        # Get sentiment prediction
        predicted_class, confidence = sentiment_api_inference(request.tweet)
        
        # Prepare response
        response = SentimentResponse(
            sentiment=predicted_class,
            confidence_score=float(confidence)
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Example usage:
# {"tweet": "I am happy!"}
