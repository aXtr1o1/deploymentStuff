import torch
from flask import Flask, request, jsonify
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
with open('apiKey.bin', 'r') as f:
    HUGGINGFACE_API_KEY = f.read().strip()

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-3.2-1B', 
    token=HUGGINGFACE_API_KEY
)
tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Llama-3.2-1B', 
    token=HUGGINGFACE_API_KEY
)

model.resize_token_embeddings(len(tokenizer))
model.load_adapter('Pragades/llama3.2-mini-QLoRA')

pipe = pipeline(
    'text-generation', 
    model=model, 
    tokenizer=tokenizer, 
    device=0 if (device == "mps" or device == "cuda") else -1
)

conversation_history = []
def manage_conversation_history(new_entry, max_length=1024):
    conversation_history.append(new_entry)
    history_str = ' '.join(conversation_history)
    while tokenizer(history_str, return_tensors='pt').input_ids.size(-1) > max_length:
        conversation_history.pop(0)
        history_str = ' '.join(conversation_history)
    return history_str

def inference(prompt, max_context_length=1024):
    conversation_context = manage_conversation_history(f"User: {prompt}", max_length=max_context_length)
    formatted_prompt = (
        f"Below is a conversation between a user and an AI. Provide an appropriate follow-up question with a normal Human Resource Manager tone and gesture based on the user's message.\n\n"
        f"Conversation history:\n{conversation_context}\n"
        f"User's latest message: {prompt}\nOutput:"
    )
    
    outputs = pipe(
        formatted_prompt,
        max_new_tokens=256,
        do_sample=True,
        early_stopping=True,
        temperature=0.3,
        top_k=50,
        top_p=0.95,
        max_time=2,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=1
    )
    generated_text = outputs[0]['generated_text'][len(formatted_prompt):].strip()
    if tokenizer.eos_token in generated_text:
        return generated_text.split(tokenizer.eos_token)[0].strip()
    else:
        return generated_text.split('<|')[0].strip()

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    response = inference(prompt)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)