from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import re
import warnings
from pathlib import Path
import os
from argparse import ArgumentParser

warnings.filterwarnings("ignore")


def load_models(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        use_flash_attention_2=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    return tokenizer, model


def load_claims(claim_file):
    with open(claim_file) as f:
        claims = json.load(f)
    return claims


def build_prompt(sentence):
    template = """### Instruction: Given a sentence, please identify the head and tail entities in the sentence, and classify the relation type into one of appropriate categories; The collection of categories is: [screenwriter, has part, said to be the same as, composer, participating team, headquarters location, heritage designation, after a work by, participant, part of, performer, work location, operating system, instance of, original language of film or TV show, follows, country of citizenship, residence, architect, position held, genre, original network, main subject, sport, mountain range, publisher, manufacturer, located on terrain feature, instrument, country of origin, position played on team / speciality, developer, military branch, movement, distributor, owned by, platform, located in or next to body of water, nominated for, location, place served by transport hub, league, religion, military rank, successful candidate, operator, country, sibling, mouth of the watercourse, constellation, child, notable work, field of work, subsidiary, winner, director, crosses, member of political party, licensed to broadcast to, tributary, location of formation, spouse, sports season of league or competition, language of work or name, occupation, head of government, occupant, mother, competition class, located in the administrative territorial entity, contains administrative territorial entity, participant of, voice type, followed by, member of, father, record label, taxon rank, characters, applies to jurisdiction]
Sentence: {} ### Response:"""
    return template.format(sentence)


def post_process(response):
    relation_pattern = r"\(\w+(?:\s+\w+)*(?:,\s+\w+(?:\s+\w+)*)*\)"
    relations = re.findall(relation_pattern, response)
    relations = [
        tuple([c.strip() for c in r.strip("(").strip(")").split(",")])
        for r in relations
    ]
    relations = list(filter(lambda triplets: len(triplets) == 3, relations))
    return relations


def extract_sentence_relations(sentence, tokenizer, model):
    prompt = build_prompt(sentence)
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            input_ids,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = output[0][input_ids.shape[-1] :]
        response = tokenizer.decode(response, skip_special_tokens=True)
    response = post_process(response)
    return response


def find_math(e, list):
    for l in list:
        if re.search(r"{}".format(e), l):
            return True
    return False


def prune_relations(claim_rels, evd_rels):
    hypos = []
    retrieve_rel = []

    evd_rels = [tuple(x) for x in evd_rels]
    evd_rels = list(set(evd_rels))
    if len(claim_rels) > 0:
        for c in claim_rels:
            hypos.append(c[0])
            hypos.append(c[2])
        hypos = list(set(hypos))

        is_found = True
        len_hypos = len(hypos)
        while is_found:
            for e in evd_rels:
                if find_math(e[0].strip(), hypos):
                    hypos.append(e[2])
                    retrieve_rel.append(e)

                    hypos = list(set(hypos))
                    retrieve_rel = list(set(retrieve_rel))
            if len_hypos == len(hypos):
                is_found = False
            else:
                len_hypos = len(hypos)

    return evd_rels


def extract_evidence_relations(evd_sentences, tokenizer, model, pbar):
    evd_rels = []
    for sentence in evd_sentences:
        sent_rels = extract_sentence_relations(sentence, tokenizer, model, pbar)
        evd_rels += sent_rels
    return evd_rels


def save_results(output_file, results):
    with open(output_file, "w") as f:
        f.write(json.dumps(results, indent=2))


def extract_relation():
    args = parse_arguments()

    cwd = Path(os.path.dirname(os.path.realpath(__file__)))
    model_path = cwd.parents[1] / "models/fine-tuned models/llama-3-8b-new"
    tokenizer, model = load_models(model_path)
    claims = load_claims(cwd.parents[1] / "datasets/hover_sample_2k.json")
    for i in range(len(claims)):
        claims[i]["evd_sentences"] = sent_tokenize(claims[i]["evidence"])

    claims = claims[args.start : args.end]

    output_file = cwd.parents[1] / f"datasets/hover_2k_relations.json"

    claims_with_evd_rels = []
    step = 0
    with tqdm(total=sum([len(claim["evd_sentences"]) for claim in claims])) as pbar:
        for claim in claims:
            evd_rels = extract_evidence_relations(
                claim["claim"],
                claim["evd_sentences"],
                tokenizer=tokenizer,
                model=model,
                pbar=pbar,
            )
            claims_with_evd_rels.append({"id": claim["id"], "evd_rels": evd_rels})
            step += 1
            if step % 100 == 0:
                save_results(output_file=output_file, results=claims_with_evd_rels)
        save_results(output_file=output_file, results=claims_with_evd_rels)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()
    return args


def run_prune_relations():
    root = Path(os.path.dirname(os.path.realpath(__file__))).parents[1]
    rels_file = root / "datasets/hover_2k_relations.json"

    with open(rels_file) as f:
        rels = json.load(f)

    claims_file = root / "datasets/hover_sample_2k.json"
    with open(claims_file) as f:
        claims = json.load(f)
    id2claim = {claim["id"]: claim["claim"] for claim in claims}

    tokenizer, model = load_models(
        model_path=root / "models/fine-tuned models/llama-3-8b-new"
    )
    output_file = root / "datasets/hover_2k_pruned_relations.json"
    for i in tqdm(range(len(rels))):
        claim_rels = extract_sentence_relations(
            id2claim[rels[i]["id"]], tokenizer=tokenizer, model=model
        )
        pruned_rels = prune_relations(
            claim_rels=claim_rels, evd_rels=rels[i]["evd_rels"]
        )
        rels[i]["evd_rels"] = pruned_rels
        if (i + 1) % 100 == 0:
            with open(output_file, "w") as f:
                f.write(json.dumps(rels, indent=2))
    with open(output_file, "w") as f:
        f.write(json.dumps(rels, indent=2))


if __name__ == "__main__":
    extract_relation()
    run_prune_relations()
