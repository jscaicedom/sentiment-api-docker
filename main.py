from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

app = FastAPI()

# --- Config ---
print("Loading local AI model (google/flan-t5-base)... this happens once.")

# Create a local pipeline
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=512,
    model_kwargs={"temperature": 0}
)

# Wrap it in LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# --- Input Model ---
class TextInput(BaseModel):
    text: str

# --- LangChain Setup ---
# Simple prompt for the local model
template = """
Analyze the sentiment of this text: "{text}".
Determine if it is "Positive", "Negative", or "Neutral".
Explain why in one short sentence.

Format the output exactly like this example:
Sentiment: Positive
Explanation: The user expressed great joy.

Now analyze:
"""

prompt = PromptTemplate(template=template, input_variables=["text"])
chain = prompt | llm

# --- Endpoint ---
@app.post("/analyze-sentiment")
async def analyze_sentiment(input_data: TextInput):
    try:
        # Run the chain
        raw_response = chain.invoke({"text": input_data.text})
        
        # Simple parsing to fake a JSON structure for the user
        # (Local models sometimes struggle with strict JSON, so we help it)
        return {
            "result": {
                "raw_output": raw_response,
                "note": "Processed locally with Flan-T5-Base (Free)"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)