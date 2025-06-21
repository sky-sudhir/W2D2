import json
import random
import logging
from validator import validate_answer, load_kb

# Configure logging
logging.basicConfig(filename='run.log', level=logging.INFO, format='%(asctime)s - %(message)s', filemode='w', encoding='utf-8')

# Dummy model function
def ask_model_dummy(question, kb):
    # Check if the question is in the knowledge base
    for pair in kb:
        if pair['question'] == question:
            # 70% chance of correct answer, 30% of incorrect for in-KB questions
            if random.random() < 0.7:
                return pair['answer']
            else:
                return "I believe the answer is something different."

    # For out-of-domain questions
    return "As an AI, I am not able to answer this question."

def run():
    kb = load_kb()
    kb_questions = [pair['question'] for pair in kb]
    
    ood_questions = [
        "What is the meaning of life?",
        "What is the best color?",
        "Predict the winner of the next FIFA world cup.",
        "Who is the best programmer in the world?",
        "What is love?"
    ]
    
    all_questions = kb_questions + ood_questions
    random.shuffle(all_questions)
    
    results = {"correct": 0, "incorrect": 0, "retried": 0, "ood": 0}
    summary_details = []

    for question in all_questions:
        logging.info(f"Asking question: {question}")
        answer = ask_model_dummy(question, kb)
        logging.info(f"Model answer: {answer}")
        
        status = validate_answer(question, answer, kb)
        logging.info(f"Validation status: {status}")

        if "RETRY" in status:
            results["retried"] += 1
            logging.info(f"Retrying question: {question}")
            answer = ask_model_dummy(question, kb)
            logging.info(f"Model answer on retry: {answer}")
            status = validate_answer(question, answer, kb)
            logging.info(f"Validation status on retry: {status}")

        if status == "OK":
            results["correct"] += 1
            summary_details.append(f"- **{question}**: ✅ Correct")
        elif "differs from KB" in status:
            results["incorrect"] += 1
            summary_details.append(f"- **{question}**: ❌ Incorrect (mismatched)")
        elif "out-of-domain" in status:
            results["ood"] += 1
            summary_details.append(f"- **{question}**: ⚠️ Out-of-Domain")

    generate_summary(results, summary_details)

def generate_summary(results, summary_details):
    logging.info(f"Generating summary with details: {summary_details}")
    with open('summary.md', 'w', encoding='utf-8') as f:
        f.write("# Model Performance Summary\n\n")
        f.write("## Overall Results\n")
        f.write(f"- **Correct Answers:** {results['correct']}\n")
        f.write(f"- **Incorrect Answers (Mismatched):** {results['incorrect']}\n")
        f.write(f"- **Out-of-Domain Questions:** {results['ood']}\n")
        f.write(f"- **Total Retries:** {results['retried']}\n\n")
        f.write("## Detailed Breakdown\n")
        f.write("\n".join(summary_details))
        f.write("\n")

if __name__ == "__main__":
    run() 