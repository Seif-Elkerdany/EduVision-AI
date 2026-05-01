from pydantic import BaseModel

class STTResponse(BaseModel):
    text: str
    language: str

class ClassificationResult(BaseModel):
    text:str
    language:str
    predicted_topic: str
    score: float