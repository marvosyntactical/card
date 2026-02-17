#!/usr/bin/env python3
"""
Memory Consolidation Experiment: KV Cache -> MLP Weights

Tests whether fine-tuning transformer weights on document content allows
the model to retain information after the KV cache is cleared.

Core idea (Ramsauer + Ambrogioni):
- Attention = modern Hopfield retrieval over KV cache (short-term memory)
- MLP layers = associative memory in weights (long-term memory)
- Consolidation = gradient descent on LM loss to write associations into MLP

Usage:
    pip install torch transformers
    python kv_consolidation.py --model gpt2-medium --device cuda
    python kv_consolidation.py --model gpt2 --device cpu --steps 100 --docs 2
    python kv_consolidation.py --model gpt2-medium --device cuda --forgetting-curve
"""

import argparse, time
from dataclasses import dataclass
from typing import List, Dict
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Config:
    model_name: str = "gpt2-medium"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    consolidation_lr: float = 1e-4
    consolidation_steps: int = 200
    reg_strength: float = 0.005   # L2 toward anchor weights
    log_every: int = 50
    max_gen_tokens: int = 40


# ============================================================
# Fictional documents — model CANNOT know these from pretraining
# ============================================================

DOCUMENTS = [
    {"id": "doc_1", "title": "Zalthar Institute",
     "text": ("The Zalthar Institute was founded in 2019 by Dr. Elena Voss in "
              "Reykjavik, Iceland. It specializes in geothermal energy storage "
              "using modified basalt formations. Their flagship project, code-named "
              "Magma Mirror, achieved a record 94.7 percent energy retention rate "
              "in 2023. The institute employs exactly 342 researchers and operates "
              "with an annual budget of 89 million euros. Their chief scientist, "
              "Dr. Kenji Watanabe, previously worked at CERN on the ATLAS detector "
              "before joining Zalthar in 2021."),
     "questions": [
         {"q": "Who founded the Zalthar Institute?", "a": ["Elena Voss", "Dr. Elena Voss"]},
         {"q": "In what city is the Zalthar Institute located?", "a": ["Reykjavik"]},
         {"q": "What is the code name of the flagship project?", "a": ["Magma Mirror"]},
         {"q": "What energy retention rate did Magma Mirror achieve?", "a": ["94.7"]},
         {"q": "How many researchers work at Zalthar?", "a": ["342"]},
         {"q": "Who is the chief scientist at Zalthar?", "a": ["Kenji Watanabe"]},
     ]},
    {"id": "doc_2", "title": "Vermillion Accord",
     "text": ("The Vermillion Accord was a landmark trade agreement signed on "
              "March 14, 2022, in the city of Bruges, Belgium. It was negotiated "
              "by Ambassador Lucia Ferreira of Brazil and Minister Oleg Drankov of "
              "Kazakhstan over eleven months. The accord established a tariff-free "
              "corridor for rare earth minerals. Its projected trade volume was "
              "estimated at 23 billion US dollars per year. It was named after the "
              "Vermillion River in northern Kazakhstan."),
     "questions": [
         {"q": "When was the Vermillion Accord signed?", "a": ["March 14", "March 14, 2022"]},
         {"q": "In what city was the Vermillion Accord signed?", "a": ["Bruges"]},
         {"q": "Who was the Brazilian negotiator?", "a": ["Lucia Ferreira"]},
         {"q": "What was the projected annual trade volume?", "a": ["23 billion"]},
         {"q": "What is the Accord named after?", "a": ["Vermillion River"]},
         {"q": "Who was the Kazakh negotiator?", "a": ["Oleg Drankov"]},
     ]},
    {"id": "doc_3", "title": "Dr. Amara Okafor",
     "text": ("Dr. Amara Okafor is a Nigerian-born computational linguist who "
              "developed the Tessera framework for low-resource language translation. "
              "Born in Lagos in 1987, she earned her PhD from the University of "
              "Edinburgh in 2015. Tessera supports translation for 47 African "
              "languages using only 8000 parallel sentences per pair. Okafor holds "
              "the Babbage Chair at the University of Cape Town. Her team of 19 "
              "researchers received the Lovelace Prize in 2023."),
     "questions": [
         {"q": "What framework did Dr. Okafor develop?", "a": ["Tessera"]},
         {"q": "In what city was Okafor born?", "a": ["Lagos"]},
         {"q": "Where did Okafor earn her PhD?", "a": ["Edinburgh", "University of Edinburgh"]},
         {"q": "How many African languages does Tessera support?", "a": ["47"]},
         {"q": "What academic chair does Okafor hold?", "a": ["Babbage Chair"]},
         {"q": "What prize did her team receive?", "a": ["Lovelace Prize"]},
     ]},
    {"id": "doc_4", "title": "Crystaline Protocol",
     "text": ("The Crystaline Protocol is a quantum error correction method "
              "invented by Professor Haruto Nishida at Kyoto University in 2020. "
              "It uses a lattice of 128 entangled qubits in a truncated octahedral "
              "geometry. In benchmarks at the Riken Institute in January 2024, "
              "Crystaline achieved a logical error rate of 0.003 percent per cycle. "
              "It requires 11 millikelvin and consumes 2.7 kilowatts."),
     "questions": [
         {"q": "Who invented the Crystaline Protocol?", "a": ["Haruto Nishida", "Nishida"]},
         {"q": "At which university was Crystaline invented?", "a": ["Kyoto University"]},
         {"q": "How many qubits does Crystaline use?", "a": ["128"]},
         {"q": "What error rate did Crystaline achieve?", "a": ["0.003"]},
         {"q": "What temperature does Crystaline require?", "a": ["11 millikelvin"]},
         {"q": "Where were the benchmarks conducted?", "a": ["Riken"]},
     ]},
    {"id": "doc_5", "title": "Selenar Archipelago",
     "text": ("The Selenar Archipelago is a chain of 14 volcanic islands 740 km "
              "southeast of New Zealand. The largest island, Korrath, spans 312 "
              "square kilometers and is home to the Selenar blue gecko with 1700 "
              "remaining individuals. The islands were first charted by Captain "
              "Astrid Lindmark in 1891. The highest peak, Mount Orthel, rises "
              "2480 meters on Korrath."),
     "questions": [
         {"q": "How many islands are in the Selenar Archipelago?", "a": ["14"]},
         {"q": "What is the largest island?", "a": ["Korrath"]},
         {"q": "How large is Korrath?", "a": ["312"]},
         {"q": "What gecko species lives there?", "a": ["Selenar blue gecko"]},
         {"q": "Who first charted the archipelago?", "a": ["Astrid Lindmark"]},
         {"q": "What is the highest peak?", "a": ["Mount Orthel", "Orthel"]},
     ]},
]

# Held-out text for measuring catastrophic forgetting on general capability
GENERAL_EVAL_TEXT = (
    "The process of photosynthesis in plants converts sunlight into chemical "
    "energy through reactions in chloroplasts. During the light reactions, water "
    "molecules are split to release oxygen and generate ATP and NADPH. In the "
    "Calvin cycle, carbon dioxide is fixed into organic molecules using the "
    "energy carriers produced in the light reactions."
)


# ============================================================
# Utilities
# ============================================================

def get_mlp_params(model):
    """Return MLP parameters for any common HF transformer architecture."""
    params = []
    for name, p in model.named_parameters():
        if ".mlp." in name.lower() or ".feed_forward." in name.lower():
            params.append(p)
    if not params:  # fallback
        for name, p in model.named_parameters():
            if "embed" not in name.lower() and "ln" not in name.lower() and "norm" not in name.lower():
                params.append(p)
    return params


def compute_answer_logprob(model, tokenizer, question, answer, context=""):
    """Mean log P(answer_tokens | context + question). Higher = better."""
    if context:
        prompt = f"{context}\n\nQuestion: {question}\nAnswer: {answer}"
        q_prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
    else:
        prompt = f"Question: {question}\nAnswer: {answer}"
        q_prompt = f"Question: {question}\nAnswer:"

    toks = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = toks["input_ids"]
    q_len = len(tokenizer(q_prompt)["input_ids"])

    if q_len >= input_ids.shape[1]:
        return -100.0
    with torch.no_grad():
        logits = model(**toks).logits

    # Score only the answer tokens (teacher-forced)
    answer_logits = logits[0, q_len - 1:-1, :]
    answer_ids = input_ids[0, q_len:]
    if answer_ids.numel() == 0:
        return -100.0
    log_probs = -F.cross_entropy(answer_logits, answer_ids, reduction="none")
    return log_probs.mean().item()


def compute_perplexity(model, tokenizer, text):
    toks = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**toks, labels=toks["input_ids"])
    return torch.exp(out.loss).item()


def generate_answer(model, tokenizer, question, context="", max_new=40):
    if context:
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
    else:
        prompt = f"Question: {question}\nAnswer:"
    toks = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**toks, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, toks["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def check_answer(generated, accepted):
    gl = generated.lower()
    return any(a.lower() in gl for a in accepted)


# ============================================================
# Consolidation: the core mechanism
# ============================================================

def consolidate(model, tokenizer, doc_text, anchor_state, config, mode="mlp_only"):
    """
    Fine-tune model on document to write information into weights.
    
    mode="mlp_only":     freeze everything except MLP layers
    mode="full_finetune": update all parameters
    
    anchor_state: state dict to regularize toward (prevents catastrophic forgetting)
    """
    toks = tokenizer(doc_text, return_tensors="pt").to(config.device)
    input_ids = toks["input_ids"]

    # Freeze/unfreeze
    if mode == "mlp_only":
        for p in model.parameters():
            p.requires_grad = False
        for p in get_mlp_params(model):
            p.requires_grad = True
    else:  # full_finetune
        for p in model.parameters():
            p.requires_grad = True

    opt_params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in opt_params)
    optimizer = torch.optim.AdamW(opt_params, lr=config.consolidation_lr, weight_decay=0.0)

    stats = {"loss": [], "n_trainable": n_train}
    model.train()

    for step in range(config.consolidation_steps):
        optimizer.zero_grad()
        outputs = model(input_ids, labels=input_ids)
        lm_loss = outputs.loss

        # L2 regularization toward anchor (preserves prior consolidations)
        reg_loss = torch.tensor(0.0, device=config.device)
        if config.reg_strength > 0:
            for name, param in model.named_parameters():
                if param.requires_grad and name in anchor_state:
                    orig = anchor_state[name].to(config.device)
                    reg_loss = reg_loss + ((param - orig) ** 2).sum()

        total = lm_loss + config.reg_strength * reg_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
        optimizer.step()
        stats["loss"].append(total.item())

        if (step + 1) % config.log_every == 0 or step == 0:
            print(f"    step {step+1:4d}/{config.consolidation_steps}  "
                  f"LM={lm_loss.item():.4f}  reg={reg_loss.item():.4f}")

    model.eval()

    # Measure how much weights changed
    delta_sq, total_sq = 0.0, 0.0
    for name, param in model.named_parameters():
        if param.requires_grad and name in anchor_state:
            orig = anchor_state[name].to(config.device)
            delta_sq += ((param.data - orig) ** 2).sum().item()
            total_sq += (orig ** 2).sum().item()
    stats["rel_delta"] = (delta_sq / (total_sq + 1e-12)) ** 0.5

    for p in model.parameters():
        p.requires_grad = False
    return stats


# ============================================================
# Evaluation
# ============================================================

def evaluate_document(model, tokenizer, doc, config, with_context=False):
    context = doc["text"] if with_context else ""
    results = []
    for qa in doc["questions"]:
        q, accepted = qa["q"], qa["a"]
        lp = compute_answer_logprob(model, tokenizer, q, accepted[0], context)
        gen = generate_answer(model, tokenizer, q, context, config.max_gen_tokens)
        match = check_answer(gen, accepted)
        results.append({"q": q, "exp": accepted[0], "gen": gen[:80],
                        "lp": lp, "match": match})

    avg_lp = sum(r["lp"] for r in results) / len(results)
    match_rate = sum(r["match"] for r in results) / len(results)
    return {"questions": results, "avg_lp": avg_lp, "match_rate": match_rate}


# ============================================================
# Stream-and-Quiz Protocol
# ============================================================

def run_method(method, documents, model, tokenizer, orig_state, config):
    """Process all docs with given method, then quiz on everything."""
    print(f"\n{'='*60}\n  Method: {method}\n{'='*60}")

    # Reset to pristine weights
    model.load_state_dict({k: v.to(config.device) for k, v in orig_state.items()})
    model.eval()
    ppl_before = compute_perplexity(model, tokenizer, GENERAL_EVAL_TEXT)
    print(f"  General PPL before: {ppl_before:.2f}")

    # Consolidation phase (sequential)
    if method in ("mlp_only", "full_finetune"):
        for i, doc in enumerate(documents):
            print(f"\n  Consolidating doc {i+1}: {doc['title']} ({method})")
            # Anchor = current state, so regularization preserves prior docs
            anchor = {k: v.clone() for k, v in model.state_dict().items()}
            stats = consolidate(model, tokenizer, doc["text"], anchor, config, mode=method)
            print(f"    Relative weight delta: {stats['rel_delta']:.6f}")

    # Quiz phase
    print(f"\n  Quizzing...")
    quiz = {}
    for doc in documents:
        ctx = (method == "kv_oracle")
        res = evaluate_document(model, tokenizer, doc, config, with_context=ctx)
        quiz[doc["id"]] = res
        print(f"    {doc['title']:24s}  LP={res['avg_lp']:+.3f}  "
              f"match={res['match_rate']:.0%}")

    ppl_after = compute_perplexity(model, tokenizer, GENERAL_EVAL_TEXT)
    ppl_d = ppl_after - ppl_before
    print(f"  General PPL after: {ppl_after:.2f} (delta={ppl_d:+.2f})")

    return {"method": method, "quiz": quiz,
            "ppl_before": ppl_before, "ppl_after": ppl_after, "ppl_delta": ppl_d}


def run_forgetting_curve(method, documents, model, tokenizer, orig_state, config):
    """After each doc, quiz on ALL docs seen so far -> triangular matrix."""
    print(f"\n{'='*60}\n  Forgetting Curve: {method}\n{'='*60}")

    model.load_state_dict({k: v.to(config.device) for k, v in orig_state.items()})
    model.eval()
    curve = []

    for i, doc in enumerate(documents):
        if method in ("mlp_only", "full_finetune"):
            print(f"  Consolidating doc {i+1}: {doc['title']}")
            anchor = {k: v.clone() for k, v in model.state_dict().items()}
            consolidate(model, tokenizer, doc["text"], anchor, config, mode=method)

        row = {}
        for j in range(i + 1):
            res = evaluate_document(model, tokenizer, documents[j], config)
            row[documents[j]["id"]] = res["avg_lp"]
        curve.append(row)

        scores = "  ".join(f"d{j+1}={row[documents[j]['id']]:+.3f}"
                           for j in range(i + 1))
        print(f"    After doc {i+1}: {scores}")

    return {"method": method, "curve": curve}


# ============================================================
# Display helpers
# ============================================================

def print_summary(results, documents):
    dids = [d["id"] for d in documents]
    titles = [d["title"][:12] for d in documents]

    hdr = f"{'Method':20s}"
    for t in titles:
        hdr += f" | {t:>12s}"
    hdr += f" | {'Avg LP':>8s} | {'Match':>6s} | {'PPL d':>7s}"
    sep = "-" * len(hdr)

    print(f"\n{'='*len(hdr)}\n  SUMMARY\n{'='*len(hdr)}")
    print(hdr)
    print(sep)

    for r in results:
        q = r["quiz"]
        row = f"{r['method']:20s}"
        lps, mrs = [], []
        for did in dids:
            lps.append(q[did]["avg_lp"])
            mrs.append(q[did]["match_rate"])
            row += f" | {q[did]['avg_lp']:>+12.3f}"
        avg_lp = sum(lps) / len(lps)
        avg_mr = sum(mrs) / len(mrs)
        row += f" | {avg_lp:>+8.3f} | {avg_mr:>6.0%} | {r['ppl_delta']:>+7.2f}"
        print(row)

    print(sep)
    print("  LP = answer log-prob (higher=better)")
    print("  Match = generated text contains correct answer")
    print("  PPL d = general perplexity change (closer to 0 = less forgetting)\n")


def print_samples(results, documents, n=3):
    doc = documents[0]
    print(f"\n{'='*70}\n  SAMPLE GENERATIONS: {doc['title']}\n{'='*70}")
    for r in results:
        print(f"\n  [{r['method']}]")
        for qa in r["quiz"][doc["id"]]["questions"][:n]:
            mark = "Y" if qa["match"] else "N"
            print(f"    Q: {qa['q']}")
            print(f"    Expected: {qa['exp']}")
            print(f"    Got:      {qa['gen'][:60]}")
            print(f"    LP={qa['lp']:+.3f}  match={mark}\n")


def print_forgetting(curve_data, documents):
    print(f"\n  Forgetting Matrix: {curve_data['method']}")
    hdr = f"  {'After':>8s}"
    for d in documents:
        hdr += f" | {d['title'][:10]:>10s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for i, row in enumerate(curve_data["curve"]):
        line = f"  {'doc '+str(i+1):>8s}"
        for j, d in enumerate(documents):
            if d["id"] in row:
                line += f" | {row[d['id']]:>+10.3f}"
            else:
                line += f" | {'--':>10s}"
        print(line)
    print()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="KV->Weight consolidation experiment")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=200,
                        help="Gradient steps per document")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--reg", type=float, default=0.005,
                        help="L2 regularization strength toward anchor")
    parser.add_argument("--docs", type=int, default=None,
                        help="Number of documents (default: all 5)")
    parser.add_argument("--forgetting-curve", action="store_true",
                        help="Also compute forgetting curves")
    args = parser.parse_args()

    config = Config(model_name=args.model, consolidation_steps=args.steps,
                    consolidation_lr=args.lr, reg_strength=args.reg)
    if args.device:
        config.device = args.device

    docs = DOCUMENTS[:args.docs] if args.docs else DOCUMENTS

    print(f"Config: model={config.model_name}  device={config.device}  "
          f"steps={config.consolidation_steps}  lr={config.consolidation_lr}  "
          f"reg={config.reg_strength}  docs={len(docs)}\n")

    # Load model
    print(f"Loading {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=torch.float32
    ).to(config.device)
    model.eval()

    # Store pristine weights for resetting between methods
    orig = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    total_p = sum(p.numel() for p in model.parameters())
    mlp_p = sum(p.numel() for p in get_mlp_params(model))
    print(f"Params: total={total_p:,}  MLP={mlp_p:,} ({mlp_p/total_p:.1%})\n")

    # Run all methods
    methods = ["no_consolidation", "mlp_only", "full_finetune", "kv_oracle"]
    all_results = []
    for method in methods:
        t0 = time.time()
        r = run_method(method, docs, model, tokenizer, orig, config)
        r["time"] = time.time() - t0
        all_results.append(r)

    # Display results
    print_summary(all_results, docs)
    print_samples(all_results, docs)

    print("  TIMING")
    for r in all_results:
        print(f"    {r['method']:20s}  {r['time']:.1f}s")

    # Optional: forgetting curves
    if args.forgetting_curve:
        for m in ["mlp_only", "full_finetune"]:
            cd = run_forgetting_curve(m, docs, model, tokenizer, orig, config)
            print_forgetting(cd, docs)

    print("\nDone.")


if __name__ == "__main__":
    main()