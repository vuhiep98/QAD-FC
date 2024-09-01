from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm
from pathlib import Path


def get_root():
    import os

    root = Path(os.path.dirname(os.path.realpath(__file__)))
    return root.parents[1]


def post_process(response):
    lines = response.split("\n")
    lines = list(
        filter(
            lambda x: all(
                [
                    "CLAIM:" not in x,
                    "EVIDENCE:" not in x,
                    "MIND MAP:" not in x,
                    "Note" not in x,
                    "Please" not in x,
                ]
            ),
            lines,
        )
    )
    return "\n".join(lines)


def reorganize():
    root = get_root()
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        use_flash_attention_2=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    with open(root / "datasets/hover_sample_2k.json") as f:
        claims = json.load(f)
    output_file = root / "datasets/reorganized_evidence_for_2kclaims_hover"

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    claims_with_evidence = []
    step = 0
    for claim in tqdm(claims):
        messages = [
            {
                "role": "system",
                "content": "Given a claim and corresponding evidence, please summarize the evidence according to the claim. The summarized evidence should contain information that could helps support or refute the claim. Here is an example:",
            },
            {
                "role": "user",
                "content": """CLAIM: Skagen Painter Peder Severin Krøyer favored naturalism along with Theodor Esbern Philipsen and the artist Ossian Elgström studied with in the early 1900s.
EVIDENCE:
Ossian Elgström (1883 – 1950) was a Swedish illustrator and writer. He was a brother of writer and visual artist Anna Lenah Elgström. Elgström studied at the Royal Swedish Academy of Arts from 1906 to 1907, and then with Kristian Zahrtmann in 1907 and with Christian Krohg in 1908. He contributed to the magazines "Strix", "Söndags-Nisse" and "Puck". He collected folkloristic material from Siberia, Greenland and Lappland, which he used in his books. Among his books are "Lapska myther" from 1914, "Lappalaiset" from 1919, and "Karesuando-lapparna" from 1922.
Peder Severin Krøyer [\'peːdə \'seveʁin \'kʁojə] (23 July 1851 – 21 November 1909), professionally known as P. S. Krøyer, was a Danish painter. He is one of the best known and beloved, and the most colorful of the Skagen Painters, a community of Danish and Nordic artists who lived, gathered, or worked in Skagen, Denmark, especially during the final decades of the 19th century. Krøyer was the unofficial leader of the group.
Peder Henrik Kristian Zahrtmann, known as Kristian Zahrtmann, (31 March 1843 – 22 June 1917) was a Danish painter. He was a part of the Danish artistic generation in the late 19th century, along with Peder Severin Krøyer and Theodor Esbern Philipsen, who broke away from both the strictures of traditional Academicism and the heritage of the Golden Age of Danish Painting, in favor of naturalism and realism.
Write me a MIND MAP.""",
            },
            {
                "role": "assistant",
                "content": """MIND MAP:
Ossian Elgström:
    Education:
        School: Royal Swedish Academy of Arts
        Year: from 1906 to 1907
        Companion 1:
            Name: Kristian Zahrtmann
            Year: 1908
        Companion 2:
            Name: Christian Krohg
            Year: 1908
    Peder Severin Krøyer:
        Profession: Painter
        Nationality: Danish
        Group affiliation: Skagen Painters
    Kristian Zahrtmann:
        Artistic generation:
            Description: late 19th century
            Fellow artists:
                Name 1: Peder Severin Krøyer
                Name 2: Theodor Esbern Philipsen
            Artistic style:
                Break away from:
                    Academicism
                    Golden Age of Danish Painting
                Embraced:
                    Naturalism
                    Realism
                
###""",
            },
            {
                "role": "user",
                "content": """CLAIM: {}
EVIDENCE: 
{}
Write me a MIND MAP.""".format(
                    claim["claim"], claim["evidence"]
                ),
            },
        ]

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
            response = post_process(response)

        claims_with_evidence.append(
            {"id": claim["id"], "reorganized_evidence": response}
        )
        step += 1

        if step % 100 == 0:
            with open(output_file, "w") as f:
                f.write(json.dumps(claims_with_evidence, indent=2))

    with open(output_file, "w") as f:
        f.write(json.dumps(claims_with_evidence, indent=2))


if __name__ == "__main__":
    reorganize()
