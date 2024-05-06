from fastapi import FastAPI, Query, Body
from enum import Enum
import uvicorn
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from pathlib import Path
import uvicorn
import os
from pydantic import BaseModel
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI


app = FastAPI()
class CustomTone(BaseModel):
    custom_tone: str

model = ChatOpenAI(model="gpt-3.5-turbo",api_key="YOUR-OPENAI-API-KEY")


prompt = ChatPromptTemplate.from_template("""
You are a content writer. You task is to generate content based on the platform that user is willing to post on, the topic of the post and the post tones.

Write a {length} post for my {platform_name} profile on the topic '{topic}'. The post should be written in a {tone} tone and be well-structured, informative, and engaging for the {platform_name} audience. Please provide the post in a format that is optimized for the {platform_name} platform, including any relevant hashtags, mentions, or links. The post should be ready to be directly published on my {platform_name} profile.
""")
output_parser = StrOutputParser()


@app.post("/generate_content")
async def generate_content_api(
    platform_name: str = Query(...),
    topic: str = Query(...),
    tone: str = Query(..., enum=["Funny posts", "Informative posts", "Inspirational posts", "Personal posts", "Conversational posts", "Formal posts", "Engaging posts", "Visual posts", "Storytelling posts", "Thought-provoking posts", "Other"]),
    length: str = Query(..., enum=["short", "medium", "long"]),
    custom_tone: CustomTone = Body(None)
):
    if tone == "Other" and custom_tone:
        tone = custom_tone.custom_tone

    chain = prompt | model | output_parser
    response = chain.invoke({"platform_name": platform_name, "topic": topic,"tone":tone,"length":length})
    return {"content": response}

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)