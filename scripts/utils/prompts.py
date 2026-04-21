"""Unified CoT and no-CoT prompts per language and dataset."""
from __future__ import annotations

from typing import Dict


COT_INSTRUCTIONS: Dict[str, str] = {
    "en": "Think step by step. Number each reasoning step (Step 1:, Step 2:, ...). At the end, output exactly one line: 'Answer: <choice>'.",
    "zh": "请逐步思考。依次为每个推理步骤编号（步骤1：、步骤2：、...）。最后仅输出一行：“答案：<选项>”。",
    "ja": "ステップごとに考えてください。各推論ステップに番号を付けてください（ステップ1：、ステップ2：、...）。最後に一行だけ：『答え：<選択肢>』と出力してください。",
    "fr": "Réfléchissez étape par étape. Numérotez chaque étape (Étape 1 :, Étape 2 :, ...). À la fin, affichez exactement une ligne : « Réponse : <choix> ».",
}

DIRECT_INSTRUCTIONS: Dict[str, str] = {
    "en": "Answer with a single choice only. Output exactly: 'Answer: <choice>' and nothing else.",
    "zh": "只输出一个选项。仅输出：“答案：<选项>”，不要输出其他内容。",
    "ja": "選択肢のみ出力してください。『答え：<選択肢>』だけを出力し、それ以外は書かないでください。",
    "fr": "Répondez par un seul choix. Affichez uniquement : « Réponse : <choix> » et rien d'autre.",
}


def _fmt_xcopa(item: Dict, lang: str) -> str:
    q = item["premise"]
    c1 = item["choice1"]
    c2 = item["choice2"]
    typ = item.get("question", "cause")
    if lang == "en":
        connector = "What was the cause?" if typ == "cause" else "What happened as a result?"
        return f"Premise: {q}\n{connector}\nChoice A: {c1}\nChoice B: {c2}\nSelect A or B."
    if lang == "zh":
        connector = "原因是什么？" if typ == "cause" else "结果是什么？"
        return f"前提：{q}\n{connector}\n选项A：{c1}\n选项B：{c2}\n请选择A或B。"
    if lang == "ja":
        connector = "原因は何ですか？" if typ == "cause" else "結果は何ですか？"
        return f"前提：{q}\n{connector}\n選択肢A：{c1}\n選択肢B：{c2}\nAまたはBを選んでください。"
    connector = "Quelle est la cause ?" if typ == "cause" else "Quel est le résultat ?"
    return f"Prémisse : {q}\n{connector}\nChoix A : {c1}\nChoix B : {c2}\nChoisissez A ou B."


def _fmt_xstorycloze(item: Dict, lang: str) -> str:
    ctx = " ".join(item[f"input_sentence_{i}"] for i in range(1, 5))
    a = item["sentence_quiz1"]
    b = item["sentence_quiz2"]
    if lang == "en":
        return f"Story: {ctx}\nWhich ending is more likely?\nChoice A: {a}\nChoice B: {b}\nSelect A or B."
    if lang == "zh":
        return f"故事：{ctx}\n哪个结尾更可能？\n选项A：{a}\n选项B：{b}\n请选择A或B。"
    if lang == "ja":
        return f"物語：{ctx}\nどちらの結末がより妥当ですか？\n選択肢A：{a}\n選択肢B：{b}\nAまたはBを選んでください。"
    return f"Histoire : {ctx}\nQuelle fin est la plus probable ?\nChoix A : {a}\nChoix B : {b}\nChoisissez A ou B."


def _fmt_mgsm(item: Dict, lang: str) -> str:
    q = item.get("question") or item.get("question_" + lang) or ""
    if lang == "en":
        return f"Problem: {q}\nSolve and give the final numeric answer."
    if lang == "zh":
        return f"题目：{q}\n请解答，并给出最终的数值答案。"
    if lang == "ja":
        return f"問題：{q}\n解いて最終的な数値を答えてください。"
    return f"Problème : {q}\nRésolvez et donnez la réponse numérique finale."


def _fmt_belebele(item: Dict, lang: str) -> str:
    passage = item["flores_passage"]
    q = item["question"]
    mc = [item[f"mc_answer{i}"] for i in range(1, 5)]
    labels = ["A", "B", "C", "D"]
    choices = "\n".join(f"{labels[i]}: {mc[i]}" for i in range(4))
    if lang == "en":
        return f"Passage: {passage}\nQuestion: {q}\n{choices}\nSelect A, B, C, or D."
    if lang == "zh":
        return f"文章：{passage}\n问题：{q}\n{choices}\n请选择A、B、C或D。"
    if lang == "ja":
        return f"本文：{passage}\n問題：{q}\n{choices}\nA、B、C、Dから選んでください。"
    return f"Passage : {passage}\nQuestion : {q}\n{choices}\nChoisissez A, B, C ou D."


FORMATTERS = {
    "xcopa": _fmt_xcopa,
    "xstorycloze": _fmt_xstorycloze,
    "mgsm": _fmt_mgsm,
    "belebele": _fmt_belebele,
}


def build_prompt(dataset: str, item: Dict, lang: str, cot: bool = True) -> str:
    body = FORMATTERS[dataset](item, lang)
    instr = COT_INSTRUCTIONS[lang] if cot else DIRECT_INSTRUCTIONS[lang]
    return f"{body}\n\n{instr}"
