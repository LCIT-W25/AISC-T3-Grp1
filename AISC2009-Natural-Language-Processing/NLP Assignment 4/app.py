from flask import Flask, request, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

app = Flask(__name__)

# === Load base GPT-2 and LoRA-adapted model ===
base_model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Path to your LoRA adapter checkpoint
peft_model_path = "C:/Users/User/Downloads/NLP ASS 4/results/checkpoint-750"
model = PeftModel.from_pretrained(base_model, peft_model_path)
model.eval()

# === Flask Route ===
@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    if request.method == "POST":
        user_input = request.form["user_input"]
        input_text = f"User: {user_input}\nBot:"

        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        # Generate response
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=150,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        response = decoded.split("Bot:")[-1].strip()

    return render_template("index.html", response=response)

# === Run Flask App ===
if __name__ == "__main__":
    app.run(debug=True)