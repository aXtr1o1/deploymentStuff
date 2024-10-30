from flask import Flask, request, jsonify
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

f = open('apiKey.bin', 'r')
apiKey = f.read()
HUGGINGFACE_API_KEY = apiKey
f.close()

model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-3.2-1B', 
    use_auth_token=HUGGINGFACE_API_KEY
)
tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Llama-3.2-1B', 
    use_auth_token=HUGGINGFACE_API_KEY
)
model.resize_token_embeddings(128258)
model.load_adapter('Pragades/llama3.1-mini-QLoRA')

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)

def get_question_substring(text):
    question_words = ['What', 'Why', 'How', 'When', 'Where', 'Which', 'Who', 'Whose', 'Whom', 'Can']
    words = text.split()
    for i, word in enumerate(words):
        if word in question_words:
            return ' '.join(words[i:])
    return text

def inference(prompt, ques):
    formatted_prompt = f"Below is a conversation between a human and an AI agent. Provide a follow-up question based on the input. Previous Question: {ques} and\n\nUser response: {prompt}\n If the answers are correct, provide a follow-up question:"
    outputs = pipe(formatted_prompt, max_new_tokens=125, do_sample=True, num_beams=1, temperature=0.3, top_k=50, top_p=0.95, max_time=100, eos_token_id=tokenizer.eos_token_id, num_return_sequences=1)
    generated_text = outputs[0]['generated_text'][len(formatted_prompt):].strip()
    result = get_question_substring(generated_text)
    if '<EOS>' in result:
        result = result[:result.index('<EOS>')]
    return result

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    ques = data.get("ques", "")
    response = inference(prompt, ques)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)