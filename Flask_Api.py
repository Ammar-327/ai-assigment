import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Optional: Set Hugging Face API token if needed
# os.environ["HF_HOME"] = "your_token_here"

model_id = "deepseek-ai/deepseek-coder-1.3b-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def generate_code(prompt):
    # Tokenize prompt input
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

    # Decode the output tokens to string
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # strip the input prompt from the beginning of output
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):]

    return generated_text.strip()

# API endpoint to handle POST request
@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    try:
        # Add a default prefix to nudge model toward code output
        prompt = f"# Python code\n{prompt}\n"

        generated_code = generate_code(prompt)
        return jsonify({"generated_code": generated_code})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
