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

model.resize_token_embeddings(128258)
model.load_adapter('Pragades/llama3.1-mini-QLoRA')
pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if (device == "mps" or device == "cuda") else -1)
def get_question_substring(text):
    question_words = ['What', 'Why', 'How', 'When', 'Where', 'Which', 'Who', 'Whose', 'Whom', 'Can']
    words = text.split()
    for i, word in enumerate(words):
        if word in question_words:
            return ' '.join(words[i:])
    return text

conversation_history = []
def manage_conversation_history(new_entry, max_length=1024):
    conversation_history.append(new_entry)
    history_str = ' '.join(conversation_history)
    while tokenizer(history_str, return_tensors='pt').input_ids.size(-1) > max_length:
        conversation_history.pop(0)
        history_str = ' '.join(conversation_history)
    return history_str

def inference(prompt, previous_question, max_context_length=1024):
    conversation_context = manage_conversation_history(f"User: {prompt} AI: {previous_question}", max_length=max_context_length)
    formatted_prompt = (
        f"Below is a friendly conversation between a user and an AI. The AI’s goal is to keep the conversation engaging and insightful "
        f"by asking relevant follow-up questions based on the user’s input.\n"
        f"Conversation history:\n{conversation_context}\n"
        f"User's latest message: {prompt}\n\nAI, with a normal Human Resource Manager Gesture please generate a thoughtful follow-up question:"
    )
    outputs = pipe(
        formatted_prompt,
        max_new_tokens=125,
        do_sample=True,
        num_beams=1,
        temperature=0.6,
        top_k=40,
        top_p=0.9,
        max_time=100,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=1
    )
    generated_text = outputs[0]['generated_text'][len(formatted_prompt):].strip()
    result = get_question_substring(generated_text)
    if '<EOS>' in result:
        result = result[:result.index('<EOS>')]
    return result

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    previous_question = data.get("previous_question", "")
    response = inference(prompt, previous_question)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)