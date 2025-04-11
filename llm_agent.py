from groq import Groq
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)


def make_groq_request(client, context, prompt):
    """Make a request to Groq API"""
    try:
        print(f'test of context: {context[:5]}')
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in Groq request: {str(e)}")
        raise