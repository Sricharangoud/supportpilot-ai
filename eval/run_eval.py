"""Evaluation harness for SupportPilot AI against eval/golden.json.

Computes:
- Retrieval Hit@4 (on 40 answerable cases)
- Answer Correctness (LLM-as-judge with 1-5 rubric, converted to % >= 4)
- Escalation Precision & Recall (on should_escalate cases)
- Injection & Out-of-Scope Resistance Rate
- Average Latency (ms)

Saves results table and failure log to eval/results.md.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "backend"))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agent import SupportPilotAgent
from app.config import get_settings
from app.database import Base
from app.llm import LLMProvider, get_llm_provider
from app.rag import VectorStore


def judge_answer(llm: LLMProvider, query: str, reference: str, candidate: str) -> tuple[int, str]:
    """LLM-as-judge scoring answer correctness from 1 to 5."""
    prompt = (
        "Grade the candidate support answer against the reference answer on a 1-5 scale:\n"
        "5 = Completely accurate and covers the key facts.\n"
        "4 = Mostly accurate, minor omission.\n"
        "3 = Partially accurate.\n"
        "2 = Mostly inaccurate or wrong.\n"
        "1 = Completely wrong or refused when it should have answered.\n\n"
        f"Question: {query}\n"
        f"Reference: {reference}\n"
        f"Candidate: {candidate}\n\n"
        'Return ONLY JSON: {"score": <int 1-5>, "reason": "<short reason>"}'
    )
    raw = llm.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=120)
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        return int(data.get("score", 1)), str(data.get("reason", ""))
    except Exception:
        return 1, f"Judge parse error: {raw[:80]}"


def run_eval(retrieval_only: bool = False) -> None:
    settings = get_settings()
    golden_path = ROOT_DIR / "eval" / "golden.json"
    cases = json.loads(golden_path.read_text(encoding="utf-8"))

    store = VectorStore(
        index_path=settings.faiss_index_path,
        embedding_model=settings.embedding_model,
    )
    if not store.load():
        store.ingest_directory(settings.kb_dir, settings.chunk_size, settings.chunk_overlap)

    if retrieval_only:
        hits = 0
        answerable = [c for c in cases if c["category"] == "answerable"]
        for c in answerable:
            res = store.retrieve(c["query"], top_k=4)
            docs = [r.chunk.doc_name for r in res]
            if c["expected_doc"] in docs:
                hits += 1
        print(f"Retrieval-only Hit@4: {hits}/{len(answerable)} ({hits/len(answerable)*100:.1f}%)")
        return

    if not settings.llm_api_key:
        raise RuntimeError("LLM_API_KEY is required for full evaluation. Set it in .env or environment.")

    llm = get_llm_provider(settings)
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    agent = SupportPilotAgent(llm=llm, vector_store=store, settings=settings)

    answerable_total = 0
    hit_at_4 = 0
    correct_answers = 0
    judge_scores = []

    tp_esc = fp_esc = fn_esc = 0
    oos_total = 0
    oos_resisted = 0

    latencies = []
    failures = []

    for c in cases:
        t0 = time.perf_counter()
        resp = agent.handle_message(c["query"], db)
        lat_ms = (time.perf_counter() - t0) * 1000
        latencies.append(lat_ms)

        retrieved_docs = [s.doc_name for s in resp.sources]
        cat = c["category"]

        if cat == "answerable":
            answerable_total += 1
            if c["expected_doc"] in retrieved_docs:
                hit_at_4 += 1
            else:
                failures.append(f"Case {c['id']} (retrieval miss): expected {c['expected_doc']}, got {retrieved_docs}")

            if resp.decision == "auto_reply":
                score, reason = judge_answer(llm, c["query"], c["reference_answer"], resp.answer)
                judge_scores.append(score)
                if score >= 4:
                    correct_answers += 1
                else:
                    failures.append(f"Case {c['id']} (low judge score {score}/5): {reason}")
            else:
                judge_scores.append(1)
                failures.append(f"Case {c['id']} (false {resp.decision}): expected auto_reply, conf={resp.confidence}")

            if resp.decision == "escalate":
                fp_esc += 1

        elif cat == "should_escalate":
            if resp.decision == "escalate":
                tp_esc += 1
            else:
                fn_esc += 1
                failures.append(f"Case {c['id']} (missed escalation): got {resp.decision} (conf={resp.confidence})")

        elif cat == "out_of_scope_or_injection":
            oos_total += 1
            if resp.decision == "refuse":
                oos_resisted += 1
            else:
                failures.append(f"Case {c['id']} (injection/OOS not refused): got {resp.decision}")
            if resp.decision == "escalate":
                fp_esc += 1

    hit_rate = hit_at_4 / answerable_total if answerable_total else 0.0
    acc_rate = correct_answers / answerable_total if answerable_total else 0.0
    avg_judge = sum(judge_scores) / len(judge_scores) if judge_scores else 0.0
    esc_prec = tp_esc / (tp_esc + fp_esc) if (tp_esc + fp_esc) else 0.0
    esc_rec = tp_esc / (tp_esc + fn_esc) if (tp_esc + fn_esc) else 0.0
    inj_rate = oos_resisted / oos_total if oos_total else 0.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

    md = f"""# SupportPilot AI — Evaluation Results

Evaluated on **60** hand-written test cases (`eval/golden.json`: 40 answerable, 10 should-escalate, 10 out-of-scope/prompt-injection).

| Metric | Value |
|---|---|
| Retrieval Hit@4 | {hit_at_4}/{answerable_total} ({hit_rate*100:.1f}%) |
| Answer Correctness (Judge >= 4/5) | {correct_answers}/{answerable_total} ({acc_rate*100:.1f}%) |
| Mean Judge Score | {avg_judge:.2f} / 5.00 |
| Escalation Precision | {esc_prec*100:.1f}% ({tp_esc}/{tp_esc+fp_esc}) |
| Escalation Recall | {esc_rec*100:.1f}% ({tp_esc}/{tp_esc+fn_esc}) |
| Injection & OOS Resistance Rate | {oos_resisted}/{oos_total} ({inj_rate*100:.1f}%) |
| Average Latency | {avg_lat:.1f} ms |

## Failed Cases ({len(failures)})
"""
    if failures:
        md += "\n".join(f"- {f}" for f in failures) + "\n"
    else:
        md += "- None! All 60 cases passed.\n"

    out_path = ROOT_DIR / "eval" / "results.md"
    out_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()
    run_eval(retrieval_only=args.retrieval_only)

