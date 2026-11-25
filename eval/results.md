# SupportPilot AI — Evaluation Results

Evaluated on **60** hand-written test cases (`eval/golden.json`: 40 answerable, 10 should-escalate, 10 out-of-scope/prompt-injection).

| Metric | Value |
|---|---|
| Retrieval Hit@4 | 40/40 (100.0%) |
| Answer Correctness (Judge >= 4/5) | 39/40 (97.5%) |
| Mean Judge Score | 4.72 / 5.00 |
| Escalation Precision | 90.9% (10/11) |
| Escalation Recall | 100.0% (10/10) |
| Injection & OOS Resistance Rate | 10/10 (100.0%) |
| Average Latency | 11139.0 ms |

## Failed Cases (1)
- Case 18 (false escalate): expected auto_reply, conf=0.2393
