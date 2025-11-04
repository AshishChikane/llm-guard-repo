"""
Make sure you have installed Groq:
    pip install groq
"""

from groq import Groq

from llm_guard import scan_output, scan_prompt
from llm_guard.input_scanners import Anonymize, PromptInjection, TokenLimit, Toxicity
from llm_guard.output_scanners import Deanonymize, NoRefusal, Relevance, Sensitive
from llm_guard.vault import Vault

# Hardcoded Groq API Key
GROQ_API_KEY = ""

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

vault = Vault()
input_scanners = [Anonymize(vault), Toxicity(), TokenLimit(), PromptInjection()]
output_scanners = [Deanonymize(vault), NoRefusal(), Relevance(), Sensitive()]

prompt = (
   "Draft a professional email to a customer. Name is Sarah Lee. Email is sarah@example.com " "but also reachable at s.lee@workmail.com. Phone number is 555-987-6543, " "IP address 192.168.10.25, and her account number is AC987654. " "She purchased a subscription for SuperApp Pro."
)


# Scan prompt
sanitized_prompt, results_valid, results_score = scan_prompt(input_scanners, prompt)
if any(results_valid.values()) is False:
    print(f"Prompt {prompt} is not valid, scores: {results_score}")
    exit(1)

print(f"Prompt: {sanitized_prompt}")

# Call Groq LLM (same style as OpenAI call)
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",   
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": sanitized_prompt},
    ],
    temperature=0,
    max_tokens=512,
)

response_text = response.choices[0].message.content

# Scan output
sanitized_response_text, results_valid, results_score = scan_output(
    output_scanners, sanitized_prompt, response_text
)
if not all(results_valid.values()) is True:
    print(f"Output {response_text} is not valid, scores: {results_score}")
    exit(1)

print(f"Output: {sanitized_response_text}\n")
