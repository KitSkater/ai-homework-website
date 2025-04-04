from flask import Flask, request, render_template_string
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
login("your_huggingface_token")

# Load Mistral 7B Model
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map="auto"
)

# Flask app
app = Flask(__name__)

# AI response function
def ask_mistral(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    output = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# HTML template
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Mistral AI Homework Helper</title>
    <style>
        body { font-family: Arial; padding: 30px; background: #f0f2f5; }
        .container { max-width: 600px; margin: auto; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }
        input[type=text] { width: 100%; padding: 10px; font-size: 16px; margin-top: 10px; margin-bottom: 20px; }
        input[type=submit] { padding: 10px 20px; font-size: 16px; background: #007BFF; color: white; border: none; border-radius: 8px; }
        .response { margin-top: 20px; background: #eee; padding: 10px; border-radius: 8px; white-space: pre-wrap; }
    </style>
</head>
<body>
    <div class="container">
        <h2>📘 Mistral AI Homework Helper</h2>
        <form method="POST">
            <label>Ask your homework question:</label>
            <input type="text" name="question" required>
            <input type="submit" value="Ask">
        </form>
        {% if response %}
        <div class="response"><strong>AI:</strong><br>{{ response }}</div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    response = None
    if request.method == "POST":
        question = request.form["question"]
        response = ask_mistral(question)
    return render_template_string(HTML, response=response)

if __name__ == "__main__":
    app.run(debug=True)
