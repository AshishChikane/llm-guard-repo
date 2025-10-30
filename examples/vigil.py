# ============================================================
# 🧠 LLM Guard + Groq Gemma2 Integration Server (Fixed Imports + scan_output prompt fix)
# ============================================================
 
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
 
# 🛡️ Correct llm-guard imports
from llm_guard import scan_prompt, scan_output
from llm_guard.input_scanners import Anonymize, PromptInjection, TokenLimit, Toxicity
from llm_guard.output_scanners import Deanonymize, NoRefusal, Relevance, Sensitive
from llm_guard.vault import Vault
 
# ============================================================
# 🚀 FLASK SETUP
# ============================================================
app = Flask(__name__)
CORS(app)
 
# ============================================================
# 🔧 GROQ CONFIGURATION
# ============================================================
# ⚠️ Hardcoded API key — for local testing only
GROQ_API_KEY = ""
 
client = Groq(api_key=GROQ_API_KEY)
MODEL_ID = "llama-3.3-70b-versatile"  # or any available Groq model
 
# ============================================================
# 🧩 Initialize LLM Guard Scanners
# ============================================================
vault = Vault()  # In-memory store for anonymized values
 
input_scanners = [
    Toxicity(),
    PromptInjection(),
]
 
output_scanners = [
    NoRefusal(),
    Sensitive(),
]
 
# ============================================================
# 🔹 ROUTE 1 — Input Filtering
# ============================================================
@app.route("/input-check", methods=["POST"])
def input_check():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
 
    if not prompt:
        return jsonify({"error": "Missing 'prompt'"}), 400
 
    sanitized_prompt, valid_map, score_map = scan_prompt(
        prompt=prompt,
        scanners=input_scanners,
    )
 
    return jsonify({
        "original_prompt": prompt,
        "sanitized_prompt": sanitized_prompt,
        "valid_map": valid_map,
        "score_map": score_map,
        "all_safe": all(valid_map.values()),
    })
 
# ============================================================
# 🔹 ROUTE 2 — Call Groq API
# ============================================================
@app.route("/send-to-llm", methods=["POST"])
def send_to_llm():
    data = request.get_json(force=True)
    prompt = data.get("sanitized_prompt", "")
 
    if not prompt:
        return jsonify({"error": "Missing 'sanitized_prompt'"}), 400
 
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=512,
        )
 
        response_text = response.choices[0].message.content
 
        return jsonify({
            "sent_prompt": prompt,
            "llm_response": response_text,
        })
    except Exception as e:
        return jsonify({"error": f"Groq API Error: {str(e)}"}), 500
 
# ============================================================
# 🔹 ROUTE 3 — Output Filtering
# ============================================================
import numpy as np
 
@app.route("/output-check", methods=["POST"])
def output_check():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    response_text = data.get("llm_response", "")
 
    if not response_text:
        return jsonify({"error": "Missing 'llm_response'"}), 400
 
    # ✅ Pass both prompt + output
    sanitized_output, valid_map, score_map = scan_output(
        prompt=prompt,
        output=response_text,
        scanners=output_scanners,
    )
 
    # ✅ Convert all NumPy/float32 values to Python floats
    def convert(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        else:
            return obj
 
    valid_map = convert(valid_map)
    score_map = convert(score_map)
 
    final_output = sanitized_output
 
 
    return jsonify({
        "original_output": response_text,
        "sanitized_output": sanitized_output,
        "final_output": final_output,
        "valid_map": valid_map,
        "score_map": score_map,
        "all_safe": all(valid_map.values()),
    })
 
 
# ============================================================
# 🔹 ROUTE 4 — Full Pipeline (Input → LLM → Output)
# ============================================================
@app.route("/process-all", methods=["POST"])
def process_all():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
 
    if not prompt:
        return jsonify({"error": "Missing 'prompt'"}), 400
 
    # Step 1 — Input filtering
    sanitized_prompt, input_valid_map, _ = scan_prompt(prompt=prompt, scanners=input_scanners)
 
    # Step 2 — Call Groq model
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": sanitized_prompt},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        response_text = response.choices[0].message.content
    except Exception as e:
        return jsonify({"error": f"Groq API Error: {str(e)}"}), 500
 
    # Step 3 — Output filtering (✅ pass prompt)
    sanitized_output, output_valid_map, _ = scan_output(
        prompt=sanitized_prompt,
        output=response_text,
        scanners=output_scanners,
    )
 
    final_output = sanitized_output if all(output_valid_map.values()) else "[REDACTED OUTPUT DUE TO POLICY VIOLATION]"
 
    return jsonify({
        "steps": {
            "input": {
                "original_prompt": prompt,
                "sanitized_prompt": sanitized_prompt,
                "safe": all(input_valid_map.values()),
            },
            "llm": {
                "raw_output": response_text,
            },
            "output": {
                "sanitized_output": sanitized_output,
                "final_output": final_output,
                "safe": all(output_valid_map.values()),
            },
        },
        "status": "complete",
    })
 
# ============================================================
# 🔹 HEALTH CHECK
# ============================================================
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "message": "✅ LLM Guard + Groq Integration Running",
        "model": MODEL_ID,
    })
 
# ============================================================
# 🚀 RUN FLASK APP
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)