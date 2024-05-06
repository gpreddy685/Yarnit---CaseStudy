
import streamlit as st
from openai import OpenAI
client = OpenAI(api_key="YOUR-OPENAI-API-KEY")
def generate_content(platform_name, topic, tone, length):
  completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
          {"role": "system",
          "content": """You are a content writer. You task is to generate content based on the platform that user is willing to post on, the topic of the post and the post tones."""},
          {"role": "user",
          "content": f"""Write a {length} post for my {platform_name} profile on the topic '{topic}'. The post should be written in a {tone} tone and be well-structured, informative, and engaging for the {platform_name} audience. Please provide the post in a format that is optimized for the {platform_name} platform, including any relevant hashtags, mentions, or links. The post should be ready to be directly published on my {platform_name} profile."""
          }
      ]
  )

  return completion.choices[0].message.content

st.title("Content Generator ✍️")

platform_name = st.text_input("Enter the platform name")

tone_list = [
    "Funny posts",
    "Informative posts",
    "Inspirational posts",
    "Personal posts",
    "Conversational posts",
    "Formal posts",
    "Engaging posts",
    "Visual posts",
    "Storytelling posts",
    "Thought-provoking posts",
    "Other"
]

tone = st.selectbox("Select the tone of your post", tone_list)
if tone == "Other":
    tone = st.text_input("Enter the custom tone")

length_options = ["short", "medium", "long"]
length = st.selectbox("Select the length of your post", length_options)

topic = st.text_input("Enter the topic")

if st.button("Generate Content"):
  questions = generate_content(platform_name, topic, tone_list, length)
  st.write(questions)
