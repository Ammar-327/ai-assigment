import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import os

# Setup Hugging Face API token (if needed)
# Uncomment and replace your token if you're setting it in the code
# os.environ["HF_HOME"] = "your_token_here"  # Set your token in code

# Load model and tokenizer
model_id = "deepseek-ai/deepseek-coder-1.3b-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Use CPU or GPU (if supported)
device = torch.device("cpu")  # Change to "dml" or "opencl" if they work with your GPU
model = model.to(device)

# Function to generate code based on input prompt
def generate_code(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Gradio UI
gr.Interface(
    fn=generate_code,
    inputs=gr.Textbox(lines=6, label="Enter Your Prompt"),
    outputs=gr.Textbox(lines=20, label="Generated Python Code"),
    title="DeepSeek Code Generator",
    description="Generate Python code with inline comments based on your prompt.",
    theme="default"
).launch()
