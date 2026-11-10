#!/usr/bin/env python3
"""Majority-vote rule for WheelView AI + human labels.

Each human labeler has one vote. The 3-model consensus counts as one vote
('ai'). A label proceeds when at least two voters agree and they form a
strict majority. An AI-vs-single-human disagreement is a tie: the human is
not overridden. The AI vote is never stored as a human label.
"""
from __future__ import annotations

from collections import Counter
from typing import List, Optional, Sequence, Tuple


def majority_outcome(
    human_votes: Sequence[Tuple[str, str]],
    ai_value: Optional[str],
) -> dict:
    voters: List[dict] = []
    seen = set()
    for name, val in human_votes:
        if not name or val in (None, "", []):
            continue
        key = str(name).strip().lower()
        if key in seen or key in ("deploytest", "assistant", "auto", "model", "ai"):
            continue
        seen.add(key)
        voters.append({"id": str(name), "type": "human", "value": str(val)})
    if ai_value not in (None, "", []):
        voters.append({"id": "ai", "type": "ai", "value": str(ai_value)})
    n = len(voters)
    empty = {
        "outcome": "no_votes",
        "passes": False,
        "majority_value": None,
        "basis": None,
        "n_agree": 0,
        "n_votes": n,
        "voters": voters,
        "agreeing_voters": [],
    }
    if n == 0:
        return empty
    counts: Counter = Counter(v["value"] for v in voters)
    top_val, top_n = counts.most_common(1)[0]
    tied_tops = [k for k, c in counts.items() if c == top_n]
    is_majority = top_n > (n / 2.0) and len(tied_tops) == 1
    agreeing = [v for v in voters if v["value"] == top_val]
    if top_n >= 2 and is_majority:
        n_h = sum(1 for v in agreeing if v["type"] == "human")
        n_a = sum(1 for v in agreeing if v["type"] == "ai")
        if n_h >= 2 and n_a:
            basis = "human+human+ai"
        elif n_h >= 2:
            basis = "human+human"
        elif n_h >= 1 and n_a:
            basis = "ai+human"
        else:
            basis = "other"
        return {
            "outcome": "majority_pass",
            "passes": True,
            "majority_value": top_val,
            "basis": basis,
            "n_agree": top_n,
            "n_votes": n,
            "voters": voters,
            "agreeing_voters": [v["id"] for v in agreeing],
        }
    if n >= 2 and (top_n * 2 == n or len(tied_tops) > 1):
        return {
            "outcome": "tie",
            "passes": False,
            "majority_value": None,
            "basis": None,
            "n_agree": top_n,
            "n_votes": n,
            "voters": voters,
            "agreeing_voters": [v["id"] for v in agreeing] if len(tied_tops) == 1 else [],
        }
    return {
        "outcome": "no_majority",
        "passes": False,
        "majority_value": None,
        "basis": None,
        "n_agree": top_n,
        "n_votes": n,
        "voters": voters,
        "agreeing_voters": [v["id"] for v in agreeing],
    }


def ai_consensus_value(a_val, a_conf, b_val, b_conf, c_val, c_conf, threshold, classes) -> Optional[str]:
    if a_val is None or b_val is None or c_val is None:
        return None
    if a_val != b_val or a_val != c_val:
        return None
    if a_val not in classes:
        return None
    if float(a_conf or 0) < threshold or float(b_conf or 0) < threshold or float(c_conf or 0) < threshold:
        return None
    return a_val


def decide_verdict(*, ai_label, ai_vote, models_complete, human_votes, majority):
    humans = [(n, v) for n, v in human_votes if v not in (None, "", [])]
    human_values = {v for _, v in humans}

    if not models_complete and not humans:
        return "unverifiable", "send to human review", "missing_image_or_model"

    if majority.get("passes"):
        mv = majority.get("majority_value")
        basis = majority.get("basis") or ""
        if ai_label and mv == ai_label:
            return "majority_pass", "keep", f"majority {basis} agrees with the AI label"
        if ai_label and mv != ai_label:
            return "majority_pass", "drop from split", f"majority {basis} is {mv}, not the AI label {ai_label}"
        return "majority_pass", "keep", f"majority {basis} = {mv}"

    if majority.get("outcome") == "tie":
        return "tie", "send to human review", "tie (e.g. ai vs a single human); human is not overridden"

    if not humans:
        if not models_complete:
            return "unverifiable", "send to human review", "missing_image_or_model"
        if ai_vote and ai_label and ai_vote == ai_label:
            return "confirmed", "keep", "3-model consensus matches the AI label (no human voters)"
        return "model_disputed", "send to human review", "models disagree or are below threshold (no human voters)"

    if ai_label and human_values and ai_label not in human_values and len(human_values) == 1:
        return "human_conflict", "send to human review", "single human disagrees with the AI label; no majority (human not overridden)"
    if ai_label and human_values and human_values != {ai_label} and len(human_values) > 1:
        return "human_conflict", "send to human review", "human votes are split and do not form a majority with the AI"
    if ai_label and human_values == {ai_label}:
        if ai_vote == ai_label:
            return "majority_pass", "keep", "ai+human agree"
        return "model_disputed", "keep", "a human agrees with the AI label but the 3 models did not reach consensus"
    return "model_disputed", "send to human review", "no majority and models did not confirm the label"
