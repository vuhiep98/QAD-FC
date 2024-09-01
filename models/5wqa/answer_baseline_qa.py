from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import json
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import classification_report


def get_root():
    import os

    root = Path(os.path.dirname(os.path.realpath(__file__)))
    return root.parents[1]


def load_dataset(root):
    with open(root / "datasets/hover_sample_2k.json") as f:
        claims = json.load(f)
    with open(
        root / "datasets/qa_for_2kclaims_hover_Meta-Llama-3-8B-Instruct.json"
    ) as f:
        qa_list = json.load(f)
    qa_list = {c["id"]: c["qas"] for c in qa_list}
    return claims, qa_list


text_comparator = SentenceTransformer("all-MiniLM-L6-v2")


def compare_answers(answer1, answer2):
    emb1 = text_comparator.encode(answer1)
    emb2 = text_comparator.encode(answer2)
    similarity_score = text_comparator.similarity(emb1, emb2)
    return True if similarity_score >= 0.5 else False


def answer_fnc():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        use_flash_attention_2=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    root = get_root()
    claims, qa_list = load_dataset(root)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    messages = [
        {
            "role": "system",
            "content": "Answer the following question based on the evidence. The answer should be short. The answer must be 'No information' if the answer can not be derived from the evidence.",
        },
        {
            "role": "user",
            "content": "EVIDENCE:\n{}\nQUESTION: {}\nThe answer is",
        },
    ]

    ls = []
    ps = []
    for claim in tqdm(claims):
        verdict = True
        for qa in qa_list[claim["id"]]:
            messages[-1] = {
                "role": "user",
                "content": "EVIDENCE:\n{}\nQUESTION: {}\nThe answer is".format(
                    claim["evidence"], qa[0]
                ),
            }
            with torch.no_grad():
                input_ids = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                ).to(model.device)
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=512,
                    eos_token_id=terminators,
                    do_sample=False,
                )
                response = outputs[0][input_ids.shape[-1] :]
                response = tokenizer.decode(response, skip_special_tokens=True)
            verdict = verdict and text_comparator(response, qa[1])
        ls.append(1 if claim["label"] == "Support" else 0)
        ps.append(1 if verdict == True else 0)

    report = classification_report(y_true=ls, y_pred=ps, digits=2)
    print(report)


if __name__ == "__main__":
    answer_fnc()
