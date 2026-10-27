from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError

load_dotenv()

app = FastAPI()
client = OpenAI()


class Chat(BaseModel):
    message: str


@app.post("/chat")
def chat(request: Chat):
    try:
        response = client.responses.create(model="gpt-5-mini", input=request.message)

        return {"answer": response.output_text}
    except APIConnectionError:
        raise HTTPException(
            status_code=503, detail="AI service temporarily unavailable"
        )
