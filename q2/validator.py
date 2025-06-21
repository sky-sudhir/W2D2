import json

def load_kb():
    with open('kb.json', 'r') as f:
        return json.load(f)

def validate_answer(question, answer, kb):
    for pair in kb:
        if pair['question'] == question:
            if pair['answer'].lower() in answer.lower():
                return "OK"
            else:
                return "RETRY: answer differs from KB"
    return "RETRY: out-of-domain" 