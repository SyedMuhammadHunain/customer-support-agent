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
        model="gemini-flash-latest",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction="Classify if the user query is related to shipping (rates, tracking, delivery, returns) or unrelated. Output 'shipping' if related, 'unrelated' otherwise.",
    output_schema=ClassificationOutput,
)


def router(node_input: dict):
    category = node_input.get("category", "unrelated")
    # If the LLM generates a slightly different string, default to unrelated
    if category.lower() not in ["shipping", "unrelated"]:
        category = "unrelated"
    return Event(output=node_input, branch=category.lower())


faq_agent = LlmAgent(
    name="faq_agent",
    model=Gemini(
        model="gemini-flash-latest",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction="You are a customer support representative for a shipping company. Answer the user's shipping question politely and concisely.",
)


def decline(node_input: dict):
    msg = "I'm sorry, I can only help with shipping-related queries like rates, tracking, delivery, or returns."
    yield Event(
        content=types.Content(role="model", parts=[types.Part.from_text(text=msg)])
    )
    yield Event(output=msg)


root_agent = Workflow(
    name="customer_support",
    edges=[
        ("START", classifier),
        (classifier, router),
        (router, {"shipping": faq_agent, "unrelated": decline, "__DEFAULT__": decline}),
    ],
    description="Customer support agent for a shipping company.",
)

app = App(
    root_agent=root_agent,
    name="app",
)
