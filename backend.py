import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "./lora_adapter"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
tokenizer = None

@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    print("Loading AI Model on CPU (Please wait)...")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    # Load Base Model on CPU
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="cpu", 
        torch_dtype=torch.float32 # Standard float for CPU
    )

    try:
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        model = model.merge_and_unload()
        print("Success: LoRA Adapter loaded!")
    except Exception as e:
        print(f"Warning: Adapter not found. Loading standard model. Error: {e}")
        model = base_model
        
    model.eval()

class RequestData(BaseModel):
    text: str 

@app.post("/generate")
async def generate(data: RequestData):
    if not model:
        raise HTTPException(status_code=500, detail="Model loading...")

    prompt = f"User: {data.text}\nAssistant:"
    
    # NO .to("cuda") HERE!
    inputs = tokenizer(prompt, return_tensors="pt") 

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "Assistant:" in full_output:
        answer = full_output.split("Assistant:")[-1].strip()
    else:
        answer = full_output

    return {"response": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)