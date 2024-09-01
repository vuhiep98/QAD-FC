from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def load_data(claim_file):
    with open(claim_file) as f:
        claims = json.load(f)
    return claims


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        use_flash_attention_2=True,
    )
    return tokenizer, model


def build_messages(claim, tokenizer):
    messages = [
        {
            "role": "system",
            "content": """As a journalist, generate relevant questions and corresponding answers based on the information provided by the given claim that help verifying all the facts mentioned in the claim. The answers must be generated from the information within the given claim. Here are some examples:""",
        },
        {
            "role": "assistant",
            "content": """Claim: The actress who played Cindy Campbell in a japanese horror film grew up in Washington.
    Relevant questions and answers:
    Q: Who did the actress grew up in Washington play in a japanese horror film?
    A: Cindy Campbell

    Q: Where did the actress who played Cindy Campbell in a japanese horror film grow up?
    A: Washington

    Q: What kind of movie did he actress who grew up in Washinton played Cindy Campell in?
    A: A Japanese horror movie 
    ###
    Claim: The director of the silent film The Italian Straw Hat, also wrote the film. The writer and director was born before Emilio Fernández.
    Relevant questions and answers:
    Q: Did the director of the silent file The Italian Straw Hat also wrote the film?
    A: Yes

    Q: Was the writer of the silent film The Italian Straw Hat born before Emilio Fernández?
    A: Yes

    Q: Was the director of the silent film The Italian Straw Hat born before Emilio Fernández?
    A: Yes
    ###
    Claim: This particular opera is by Christoph Willibald Gluck, not Der Barbier von Bagdad. That opera was written in 1946 by the man who provided a libretto in Chimène.
    Relevant questions and answers:
    Q: Who wrote the opera in 1946?
    A: Christoph Willibald Gluck

    Q: Did Der Barbier von Bagdad wrote the opera in 1946?
    A: No

    Q: Did Christoph Willibald Gluck provide a libretto in Chimène?
    A: Yes
    ###""",
        },
        {
            "role": "user",
            "content": f"Claim: {claim}\nRelevant questions:",
        },
    ]

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    return terminators, messages


def post_process(response):
    lines = response.split("\n")
    lines = list(
        filter(lambda line: line.startswith("Q: ") or line.startswith("A: "), lines)
    )
    lines = [line[3:] for line in lines]
    pairs = [lines[i : i + 2] for i in range(0, len(lines), 2)]
    pairs = list(filter(lambda pair: len(pair) == 2, pairs))
    return pairs


def reasoning(claim, tokenizer, model):
    terminators, messages = build_messages(claim=claim, tokenizer=tokenizer)
    with torch.no_grad():
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        outputs = model.generate(
            input_ids,
            max_new_tokens=128,
            eos_token_id=terminators,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = outputs[0][input_ids.shape[-1] :]
        response = tokenizer.decode(response, skip_special_tokens=True)
    qa_list = post_process(response=response)
    return qa_list


def main():
    os.environ["CUDA_VISIBLE_DEVICE"] = "0"
    root_dir = Path(__file__).parents[1]

    claims = load_data(claim_file=root_dir.parent / "datasets/hover_sample_2k.json")
    tokenizer, model = load_model(model_path="meta-llama/Meta-Llama-3-8B-Instruct")

    output_file = (
        root_dir / "datasets/qa_for_2kclaims_hover_Meta-Llama-3-8B-Instruct.json"
    )
    generated_qa = []
    with tqdm(total=len(claims)) as pbar:
        for i, claim in enumerate(claims):
            qa_list = reasoning(claim=claim["claim"], tokenizer=tokenizer, model=model)
            generated_qa.append({"id": claim["id"], "qa": qa_list})
            pbar.update(1)
            if (i + 1) % 100 == 0:
                print(f"Save results at step {i+1}")
                with open(output_file, "w") as f:
                    f.write(json.dumps(generated_qa, indent=2))
    with open(output_file, "w") as f:
        f.write(json.dumps(generated_qa, indent=2))


if __name__ == "__main__":
    main()
