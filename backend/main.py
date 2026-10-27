from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Chat(BaseModel):
    message: str

@app.post("/chat")
def chat(request: Chat):
    return {"answer": f"You said {request.message}"}