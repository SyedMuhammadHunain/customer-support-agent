# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import google.auth
from pydantic import BaseModel, Field

from google.adk.agents import LlmAgent
from google.adk.workflow import Workflow
from google.adk.events.event import Event
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types

# User has provided GEMINI_API_KEY and requested not to use Vertex AI
# os.environ["GEMINI_API_KEY"] = "..." is expected to be loaded by dotenv or shell


class ClassificationOutput(BaseModel):
    category: str = Field(description="Must be 'shipping' or 'unrelated'")


classifier = LlmAgent(
    name="classifier",
    model=Gemini(
        model="gemini-flash-lite-latest",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction="Classify if the user query is related to shipping (rates, tracking, delivery, returns) or unrelated. Output 'shipping' if related, 'unrelated' otherwise.",
    output_schema=ClassificationOutput,
)


from google.adk.agents.context import Context


def router(ctx: Context, node_input: dict):
    category = node_input.get("category", "unrelated")
    # If the LLM generates a slightly different string, default to unrelated
    if category.lower() not in ["shipping", "unrelated"]:
        category = "unrelated"

    # Pass the user's latest message to the next node
    user_prompt = ""
    if ctx.user_content and ctx.user_content.parts:
        user_prompt = ctx.user_content.parts[0].text
    return Event(output=user_prompt, route=category.lower())


faq_agent = LlmAgent(
    name="faq_agent",
    model=Gemini(
        model="gemini-flash-lite-latest",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are a super enthusiastic and playful customer support representative for a shipping company! 🚢✨ "
        "Answer the user's shipping questions with high energy and emojis. "
        "Make sure to always excitedly highlight that we offer FREE SHIPPING on all orders over $50! 🎉📦 "
        "Keep your answers polite, helpful, and concise."
    ),
)


def decline(node_input: dict):
    msg = "I'm sorry, I can only help with shipping-related queries like rates, tracking, delivery, or returns."
    yield Event(
        content=types.Content(role="model", parts=[types.Part.from_text(text=msg)])
    )
    yield Event(output=msg)


from typing import Any
def extract_text(node_input: Any) -> str:
    print("EXTRACT TEXT INPUT:", node_input)
    if isinstance(node_input, str):
        return node_input
    if hasattr(node_input, "parts") and node_input.parts:
        return node_input.parts[0].text
    return str(node_input)

root_agent = Workflow(
    name="customer_support",
    edges=[
        ("START", classifier),
        (classifier, router),
        (router, {
            "shipping": faq_agent,
            "__DEFAULT__": decline
        }),
        (faq_agent, extract_text)
    ],
    description="Customer support agent for a shipping company.",
)

app = App(
    root_agent=root_agent,
    name="app",
)
