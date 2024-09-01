from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import warnings
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")


def get_root():
    import os

    root = Path(os.path.dirname(os.path.realpath(__file__)))
    return root.parents[1]


def create_dict(json_file, data_type):
    with open(json_file) as f:
        data = json.load(f)
    return {d["id"]: d[data_type] for d in data}


def load_models(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        use_flash_attention_2=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    return tokenizer, model


def load_data(claim_file, qa_file, evd_file, rels_file):
    id2qa = create_dict(qa_file, "qas")
    id2evd = create_dict(evd_file, "reorganized_evidence")
    id2rels = create_dict(rels_file, "evd_rels")

    with open(claim_file) as f:
        claims = json.load(f)
    claims = [
        {
            **claim,
            "qas": id2qa[claim["id"]],
            "evidence": id2evd[claim["id"]],
            "evd_rels": id2rels[claim["id"]],
        }
        for claim in claims
    ]
    return claims


def build_prompt(question, evidence, rels):
    rels = " ".join([str(tuple(r)) for r in rels])
    context = f"EVIDENCE:\n{evidence}\nRELATIONS FROM THE EVIDENCE:\n{rels}"
    messages = [
        {
            "role": "system",
            "content": """Answer the following question based on the given context. The answer should be short. The answer must be "No information" if the answer can not be derived from the context.""",
        },
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\nQUESTION: {question}\nThe answer is",
        },
    ]
    return messages


text_comparator = SentenceTransformer("all-MiniLM-L6-v2")


def compare_answers(answer1, answer2):
    emb1 = text_comparator.encode(answer1)
    emb2 = text_comparator.encode(answer2)
    similarity_score = text_comparator.similarity(emb1, emb2)
    return True if similarity_score >= 0.5 else False


def answer(messages, tokenizer, model):
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    with torch.no_grad():
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        outputs = model.generate(
            input_ids,
            max_new_tokens=32,
            eos_token_id=terminators,
            do_sample=False,
        )
        response = outputs[0][input_ids.shape[-1] :]
        response = tokenizer.decode(response, skip_special_tokens=True)
    return response


def main():
    root = get_root()
    claims = load_data(
        claim_file=root / "datasets/hover_sample_2k.json",
        qa_file=root / "datasets/qa_for_2kclaims_hover.json",
        evd_file=root / "datasets/reorganized_evidence_for_2kclaims_hover.json",
        rels_file=root / "datasets/hover_2k_pruned_relations.json",
    )
    tokenizer, model = load_models("meta-llama/Meta-Llama-3-8B-Instruct")

    ls = []
    ps = []
    for claim in tqdm(claims):
        verdict = True
        for qa in claim["qas"]:
            msg = build_prompt(qa[0], claim["evidence"], claim["evd_rels"])
            ans = answer(msg, tokenizer=tokenizer, model=model)
            verdict = verdict and compare_answers(ans, qa[1])

        ls.append(1 if claim["label"] == "Support" else 0)
        ps.append(1 if verdict == True else False)

    report = classification_report(y_true=ls, y_pred=ps, digits=2)
    print(report)


if __name__ == "__main__":
    main()
