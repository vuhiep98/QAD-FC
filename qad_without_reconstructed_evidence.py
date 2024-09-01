import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.metrics import classification_report

template = "Answer the following question based on the evidence. The answer should be short. The answer must be 'No information' if the answer can not be derived from the evidence.\nEVIDENCE:\n{}\nQUESTION: {}\nThe answer is"


def build_prompt(evidence, question):
    return template.format(evidence, question)


text_comparator = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")


def text_generate(prompt):
    with torch.no_grad():
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(
            prompt, max_new_tokens=128, do_sample=False, temperature=0.0
        )
        response = outputs[0][inputs.shape[-1] :]
        text = tokenizer.decode(response, skip_special_token=True)
    return text


def text_similarity(text_1, text_2):
    emb_1 = text_comparator.encode(text_1)
    emb_2 = text_comparator.encode(text_2)
    similarity_score = text_comparator.similarity(emb_1, emb_2)
    return True if similarity_score >= 0.5 else False


with open("data/5WQA_all_claims_with_evidence.json") as f:
    claims = json.load(f)

preds = []
labels = []
for claim in claims:
    labels.append(1 if claim["label"] == "Support" else 0)
    verdict = True
    for qa in claim["qas"]:
        prompt = build_prompt(claim["evidence"], qa[0])
        answer = text_generate()
        similarity = text_similarity(qa[1], answer)
        verdict = verdict and similarity
    preds.append(1 if verdict == True else False)

report = classification_report(y_true=labels, y_pred=preds, digits=2)
print(report)
