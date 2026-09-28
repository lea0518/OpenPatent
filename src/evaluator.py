# -*- coding: utf-8 -*-
"""
【阶段3】评估 agent —— LLM-as-a-Judge 专利质量裁判器

定位：独立的离线评估模块，不接入生成流程（生成闭环留到阶段4）。
作用：用大模型从「语义层面」评估专利三部分质量，补上 term_metrics.py 正则法
      测不了的同义词漂移、特征支撑等问题，输出结构化评分 + 理由。

评估维度（刻意对齐阶段1/2 的改进点，使评分能印证撰写 agent 的改善）：
  1. 术语一致性 term_consistency   —— 同一概念是否全程同一措辞（含同义漂移）
  2. 权利要求-说明书支撑 claim_support —— 权利要求每个特征能否在说明书找到依据
  3. 无自造术语 no_fabricated       —— 是否出现说明书/交底书都没有的部件名
  4. 流畅规范 fluency               —— 专利语言是否规范流畅

三种评估模式及实测结论（D1，裁判 Qwen2.5-72B）：
  - score   绝对打分：区分度不足，三阶段总分均为 4.5，无法体现差异。保留作方法探索的对照。
  - pair    成对比较：正反问询消除位置偏见后仍多判"相当"，灵敏度不够。同上保留作对照。
  - extract 问题提取：成功。用"问题条数"量化，stage0→1→2 = 22→6→2，与正则法交叉验证一致。
            结论：LLM 适合"逐条挑错"（信息抽取）而非"打抽象分数"。extract 为阶段3 主评估方法。

避免自我偏袒（self-preference bias）：默认用与生成模型不同的裁判模型。
  优先级：EVAL_MODEL 环境变量 > SiliconFlow 的 Qwen(异厂商) > 回退生成模型(会告警)。

用法：
  python evaluator.py <交底书docx> <说明书docx> <摘要docx> <权利要求docx> [--glossary 术语表txt] [--tag 标签]
输出：
  打印评分 + 写 eval_result_<tag>.json / eval_result_<tag>.txt 到说明书所在目录
"""
import os
import sys
import json
import re
import argparse
import logging
from docx import Document
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def read_docx(path):
    """读取 docx 全文文字"""
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def read_txt(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def pick_judge_model():
    """
    选择裁判模型，尽量与生成模型（deepseek-chat）不同，避免自我偏袒。
    返回 (client, model_name, warn)。warn 非空表示存在自我偏袒风险，需在报告中标注。
    """
    # 1) 用户显式指定评估模型
    eval_model = os.getenv("EVAL_MODEL")
    eval_base = os.getenv("EVAL_API_BASE")
    eval_key = os.getenv("EVAL_API_KEY")
    if eval_model and eval_base and eval_key:
        return OpenAI(api_key=eval_key, base_url=eval_base), eval_model, ""

    # 2) 有 SiliconFlow key，用异厂商的 Qwen 当裁判（推荐，避免自我偏袒）
    sf_key = os.getenv("SILICONFLOW_API_KEY")
    if sf_key:
        return (
            OpenAI(api_key=sf_key, base_url="https://api.siliconflow.cn/v1"),
            "Qwen/Qwen2.5-72B-Instruct",
            "",
        )

    # 3) 回退到生成模型（deepseek），存在自我偏袒风险
    gen_key = os.getenv("LLM_API_KEY")
    gen_base = os.getenv("LLM_API_BASE")
    gen_model = os.getenv("LLM_MODEL")
    return (
        OpenAI(api_key=gen_key, base_url=gen_base),
        gen_model,
        "⚠️ 裁判模型与生成模型相同，存在自我偏袒(self-preference bias)风险，结论需谨慎；建议配置 EVAL_MODEL 或 SILICONFLOW_API_KEY。",
    )


def build_prompt(tech, spec, abst, claim, glossary):
    """构造评审 prompt，要求模型输出结构化 JSON。"""
    glossary_block = f"\n### 术语表（若提供，作为一致性判断参照）：\n{glossary}\n" if glossary else ""
    return f"""你是一名资深专利审查员。请严格、客观地评审下面这份专利文档的三个部分（说明书、摘要、权利要求），从四个维度打分。

### 评分规则：
- 每个维度打 1-5 分整数（1=很差，3=合格，5=优秀），并给出简短理由。
- 必须指出具体问题项（引用原文词语），不要泛泛而谈。
- 只依据给定材料判断，不要臆测材料之外的内容。

### 四个维度：
1. term_consistency（术语一致性）：同一部件/概念在三部分中是否全程使用完全一致的措辞？是否存在同义词漂移（如"语音交互"vs"语音模块"）、简称不一致（如"8bit .GGUF"被简写成".GGUF"）？
2. claim_support（权利要求-说明书支撑）：权利要求中出现的每个技术特征，是否都能在说明书中找到对应描述？列出无支撑的特征。
3. no_fabricated（无自造术语）：是否出现了说明书和交底书中都不存在的、模型自行发明的部件/模块名？列出可疑项。
4. fluency（流畅规范）：专利语言是否规范、通顺、符合专利撰写习惯？

### 交底书（技术原始输入）：
{tech}
{glossary_block}
### 说明书：
{spec}

### 摘要：
{abst}

### 权利要求：
{claim}

### 输出要求：
只输出一个 JSON 对象，不要输出任何多余文字。所有列表元素必须是合法的 JSON 字符串（用双引号包裹的一整段文本，元素内部不要再出现未转义的双引号；如需表达"A 与 B 不一致"，写成 "A 与 B 不一致" 这样的一个字符串）。格式如下：
{{
  "term_consistency": {{"score": 整数, "reason": "理由", "issues": ["具体问题词语", ...]}},
  "claim_support": {{"score": 整数, "reason": "理由", "unsupported_features": ["无支撑的特征", ...]}},
  "no_fabricated": {{"score": 整数, "reason": "理由", "suspects": ["可疑自造术语", ...]}},
  "fluency": {{"score": 整数, "reason": "理由"}},
  "overall": {{"score": 数字, "summary": "总体评价"}}
}}"""


def parse_json(text):
    """从模型输出中鲁棒地抽取 JSON（容忍 ```json 代码块包裹或前后有杂字）。"""
    text = text.strip()
    # 去掉 ```json ... ``` 包裹
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        # 抽取第一个 { 到最后一个 } 之间
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1:
            text = text[s:e + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(_repair_json(text))


def _repair_json(text):
    """
    降级修复模型常见的非法 JSON：数组元素里写成  "A" 与 "B" 不一致  这类
    内部含裸双引号的写法（连接词中英文皆有，甚至引号后还带裸中文）。
    策略：定位每个扁平数组 [...]，按逗号拆成元素，凡含引号的元素一律
    剥掉内部所有引号、整体重新用一对引号包裹，保证是合法 JSON 字符串。
    """
    def fix_array(m):
        body = m.group(1)
        if not body.strip():
            return "[]"
        parts = body.split(",")
        fixed_parts = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if '"' in p:
                # 剥掉内部所有引号，压缩空白，整体重新包一对引号
                inner = p.replace('"', "").strip()
                inner = re.sub(r"\s+", " ", inner)
                fixed_parts.append('"' + inner + '"')
            else:
                fixed_parts.append(p)
        return "[" + ", ".join(fixed_parts) + "]"

    # 只匹配不含嵌套中括号的扁平数组（本评估的 issues/suspects 等都是字符串列表）
    return re.sub(r"\[([^\[\]]*)\]", fix_array, text, flags=re.DOTALL)


# ============ 绝对打分（score）模式 ============
# 【实测结论】区分度不足：D1 三阶段(0/1/2)总分均为 4.5，LLM 倾向给安全高分(宽大效应)，
# 测不出细粒度差异。保留作"方法探索"的对照，不作主评估方法。
def evaluate(tech, spec, abst, claim, glossary=""):
    client, model, warn = pick_judge_model()
    prompt = build_prompt(tech, spec, abst, claim, glossary)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是严格客观的专利审查员，只输出JSON。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,  # 评估要确定性，便于复现
        stream=False,
    )
    raw = resp.choices[0].message.content
    try:
        result = parse_json(raw)
    except Exception as e:
        logging.error(f"JSON解析失败: {e}")
        result = {"_parse_error": str(e), "_raw": raw}
    result["_judge_model"] = model
    if warn:
        result["_warning"] = warn
    return result


# ============ 成对比较（pairwise）模式 ============
# 学界主流范式（MT-Bench / Chatbot Arena）。绝对打分区分度不足时改用它。
# 【实测结论】灵敏度仍不足：正反问询消除位置偏见后，stage0 vs stage2 多数维度判"相当"，
# 甚至因"泛读"误判 stage0 流畅度略高。裁判整体印象判断，抓不住细粒度术语差异。仍作对照。
# 关键：同一对正反各问一次，消除位置偏见（position bias）。

DIMS_PAIR = [
    ("term_consistency", "术语一致性：同一部件/概念是否全程一致，有无同义词漂移、简称不一致"),
    ("claim_support", "权利要求-说明书支撑：权利要求的每个技术特征能否在说明书找到依据"),
    ("no_fabricated", "无自造术语：有无说明书/交底书都不存在的自造部件名"),
    ("fluency", "流畅规范：专利语言是否规范通顺"),
]


def build_pair_prompt(tech, docA, docB, glossary):
    """构造成对比较 prompt：A、B 两份专利，逐维度判哪份更好。"""
    glossary_block = f"\n### 术语表（一致性判断参照）：\n{glossary}\n" if glossary else ""
    dim_lines = "\n".join(f"  - {k}：{desc}" for k, desc in DIMS_PAIR)
    return f"""你是一名资深专利审查员。下面是针对【同一份交底书】生成的两份专利文档 A 和 B。
请就每个维度，客观判断 A 和 B 哪一份更好，或二者相当。

### 判断维度：
{dim_lines}

### 判断规则：
- 每个维度的结论只能是 "A"（A更好）、"B"（B更好）或 "tie"（相当）。
- 必须给出简短理由，指出具体差异（引用原文词语）。
- 只依据给定材料判断，严禁臆测材料之外内容；不要因为 A/B 的先后顺序而有偏向。

### 交底书（技术原始输入）：
{tech}
{glossary_block}
### 文档 A（说明书 + 摘要 + 权利要求）：
{docA}

### 文档 B（说明书 + 摘要 + 权利要求）：
{docB}

### 输出要求：
只输出一个 JSON 对象，不要输出多余文字，格式如下：
{{
  "term_consistency": {{"winner": "A/B/tie", "reason": "理由"}},
  "claim_support": {{"winner": "A/B/tie", "reason": "理由"}},
  "no_fabricated": {{"winner": "A/B/tie", "reason": "理由"}},
  "fluency": {{"winner": "A/B/tie", "reason": "理由"}},
  "overall": {{"winner": "A/B/tie", "reason": "总体理由"}}
}}"""


def _judge_once(client, model, tech, docA, docB, glossary):
    """单次成对比较，返回各维度 winner('A'/'B'/'tie')。"""
    prompt = build_pair_prompt(tech, docA, docB, glossary)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是严格客观的专利审查员，只输出JSON。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        stream=False,
    )
    return parse_json(resp.choices[0].message.content)


def _combine(w1, w2_swapped):
    """
    合并正反两次的结论，消除位置偏见。
    w1: 正向(A=甲,B=乙)的 winner；w2_swapped: 反向(A=乙,B=甲)已换回甲乙视角的 winner。
    两次一致 -> 该结论；不一致 -> tie（说明模型分不出，判平）。
    """
    if w1 == w2_swapped:
        return w1
    return "tie"


def compare_pair(nameA, docA, nameB, docB, tech, glossary=""):
    """
    成对比较 nameA(甲) vs nameB(乙)，正反各问一次消除位置偏见。
    返回每个维度的最终胜者（以甲/乙表示）+ 两次原始结论。
    """
    client, model, warn = pick_judge_model()

    # 正向：A=甲, B=乙
    r_fwd = _judge_once(client, model, tech, docA, docB, glossary)
    # 反向：A=乙, B=甲（交换位置）
    r_bwd = _judge_once(client, model, tech, docB, docA, glossary)

    result = {"_judge_model": model, "A": nameA, "B": nameB, "dims": {}}
    if warn:
        result["_warning"] = warn

    for key, _desc in DIMS_PAIR + [("overall", "")]:
        w_fwd = r_fwd.get(key, {}).get("winner", "tie")
        # 反向里的 A 其实是乙、B 是甲，换回甲乙视角：A<->B 对调
        w_bwd_raw = r_bwd.get(key, {}).get("winner", "tie")
        w_bwd = {"A": "B", "B": "A", "tie": "tie"}.get(w_bwd_raw, "tie")
        final = _combine(w_fwd, w_bwd)
        # 把 A/B 翻译成甲(nameA)/乙(nameB)/tie
        winner_name = {"A": nameA, "B": nameB, "tie": "相当"}[final]
        result["dims"][key] = {
            "winner": winner_name,
            "consistent": w_fwd == w_bwd,  # 正反是否一致（不一致说明有位置偏见/难分）
            "forward": w_fwd,
            "backward": w_bwd,
            "reason_fwd": r_fwd.get(key, {}).get("reason", ""),
            "reason_bwd": r_bwd.get(key, {}).get("reason", ""),
        }
    return result


def format_pair_report(result):
    """格式化成对比较结果为可读文本。"""
    lines = []
    dim_names = {
        "term_consistency": "术语一致性",
        "claim_support": "权利要求-说明书支撑",
        "no_fabricated": "无自造术语",
        "fluency": "流畅规范",
        "overall": "总体",
    }
    lines.append("=" * 55)
    lines.append(f"成对比较: 甲={result['A']}  vs  乙={result['B']}")
    lines.append(f"裁判模型: {result.get('_judge_model', '?')}")
    if result.get("_warning"):
        lines.append(result["_warning"])
    lines.append("(每维度正反各问一次；正反一致才判胜负，否则判'相当')")
    lines.append("=" * 55)
    for key, d in result["dims"].items():
        flag = "" if d["consistent"] else "  ⚠正反不一致→判平"
        lines.append(f"\n【{dim_names.get(key, key)}】胜者: {d['winner']}{flag}")
        lines.append(f"  正向理由: {d['reason_fwd']}")
        lines.append(f"  反向理由: {d['reason_bwd']}")
    return "\n".join(lines)


# ============ 问题提取（extract）模式 ============
# 把 LLM 当"语义放大镜"而非"打分器"：逐条抽取可计数的具体问题，
# 用"问题条数"作为量化指标（越少越好），天然拉开阶段差距，且能与正则法交叉验证。
# 【实测结论】★成功，阶段3主评估方法：D1 问题总数 stage0→1→2 = 22→6→2，单调下降；
# 与 term_metrics 正则法结论一致，且能捕捉正则漏掉的语义问题(如"轻音乐推荐"无说明书支撑)。

def build_extract_prompt(tech, spec, abst, claim, glossary):
    glossary_block = f"\n### 术语表（一致性判断参照）：\n{glossary}\n" if glossary else ""
    return f"""你是一名极其严谨、吹毛求疵的专利审查员。请【逐条】找出下面这份专利文档中的具体问题。
不要给分、不要泛泛评价，只客观地【列举】具体问题实例，找不到就返回空列表。

### 需要逐条排查的三类问题：
1. term_drift（术语漂移）：同一个部件/概念在不同部分用了【不同】写法。逐对列出，格式："写法甲 | 写法乙 | 出现位置"。
   例如："8bit .GGUF格式 | .GGUF格式 | 说明书用全称,权利要求丢了8bit"
   ⚠️ 严禁列出写法一致的术语：若某术语在各部分写法完全相同，它【没有问题】，绝对不要列入，
   也不要写"说明书和权利要求一致"之类的说明。写法甲与写法乙必须【字面不同】，否则不得列出。
   ⚠️ 也不要把"整体与其部件"当作漂移：如"车载设备"与"车载设备的CPU"是不同概念，不算同一概念的两种写法。
   本项只统计同一概念被写成两种不同措辞的情况；找不到就返回空列表。
2. unsupported_claims（无说明书支撑的权利要求特征）：权利要求里出现、但说明书中找不到对应描述的技术特征。逐条列出该特征名及原因。
3. fabricated_terms（自造术语）：说明书、摘要或权利要求中出现，但交底书和术语表里都没有、疑似模型自行发明的部件/模块名。逐条列出。

### 排查要求：
- 必须逐字比对，不要因为"整体看起来还行"就跳过细节。
- 每一条都要引用文档中的原始词语，便于核对。

### 交底书：
{tech}
{glossary_block}
### 说明书：
{spec}

### 摘要：
{abst}

### 权利要求：
{claim}

### 输出要求：
只输出一个 JSON 对象，列表元素必须是合法 JSON 字符串（用双引号包裹的一整段文本，内部不要出现未转义的双引号）：
{{
  "term_drift": ["写法甲 与 写法乙 在xx处不一致", ...],
  "unsupported_claims": ["特征X 在说明书中无描述", ...],
  "fabricated_terms": ["自造词Y", ...]
}}"""


def _filter_same_term_drift(items):
    """剔除"写法甲与写法乙其实相同"的伪漂移条目。

    裁判有时会把各部分写法一致的术语也列进 term_drift（并附"说明书和权利要求一致"
    之类的说明），若直接计数会虚高问题总数。这里按 '|' 劈出前两段比对，字面相同即丢弃。
    返回 (保留的条目, 被丢弃的条目)。不含 '|' 的条目保守保留，交由人工复核。
    """
    kept, dropped = [], []
    for it in items:
        parts = [p.strip() for p in str(it).split("|")]
        if len(parts) >= 2 and parts[0] and parts[0] == parts[1]:
            dropped.append(it)
        else:
            kept.append(it)
    return kept, dropped


def extract_issues(tech, spec, abst, claim, glossary=""):
    """对单份文档抽取三类问题，返回含计数的结果。"""
    client, model, warn = pick_judge_model()
    prompt = build_extract_prompt(tech, spec, abst, claim, glossary)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是严格客观的专利审查员，只输出JSON。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        stream=False,
    )
    raw = resp.choices[0].message.content
    try:
        data = parse_json(raw)
    except Exception as e:
        return {"_parse_error": str(e), "_raw": raw, "_judge_model": model}

    result = {"_judge_model": model}
    if warn:
        result["_warning"] = warn
    for key in ("term_drift", "unsupported_claims", "fabricated_terms"):
        items = data.get(key, []) or []
        if key == "term_drift":
            items, dropped = _filter_same_term_drift(items)
            if dropped:
                # 裁判仍把"两处写法一致"报成漂移时，程序侧硬过滤，避免污染计数
                result["_drift_filtered"] = dropped
        result[key] = {"count": len(items), "items": items}
    result["total_issues"] = sum(result[k]["count"] for k in
                                 ("term_drift", "unsupported_claims", "fabricated_terms"))
    return result


def format_extract_report(result, tag):
    lines = []
    names = {
        "term_drift": "术语漂移",
        "unsupported_claims": "无说明书支撑的权利要求特征",
        "fabricated_terms": "自造术语",
    }
    lines.append("=" * 55)
    lines.append(f"问题提取: {tag}")
    lines.append(f"裁判模型: {result.get('_judge_model', '?')}")
    if result.get("_warning"):
        lines.append(result["_warning"])
    lines.append("=" * 55)
    if "_parse_error" in result:
        lines.append(f"⚠️ 无法解析JSON: {result['_parse_error']}")
        lines.append(result.get("_raw", ""))
        return "\n".join(lines)
    for key, name in names.items():
        d = result.get(key, {"count": 0, "items": []})
        lines.append(f"\n【{name}】共 {d['count']} 条")
        for it in d["items"]:
            lines.append(f"  - {it}")
    lines.append(f"\n>>> 问题总数: {result.get('total_issues', 0)} 条（越少越好）")
    return "\n".join(lines)


def format_report(result, tag):
    """把评分结果格式化成人类可读文本。"""
    lines = []

    def out(s):
        lines.append(s)

    out("=" * 55)
    out(f"评估标签: {tag}")
    out(f"裁判模型: {result.get('_judge_model', '?')}")
    if result.get("_warning"):
        out(result["_warning"])
    out("=" * 55)

    if "_parse_error" in result:
        out(f"⚠️ 模型输出无法解析为JSON: {result['_parse_error']}")
        out("原始输出：")
        out(result.get("_raw", ""))
        return "\n".join(lines)

    dims = [
        ("term_consistency", "术语一致性", "issues", "问题词语"),
        ("claim_support", "权利要求-说明书支撑", "unsupported_features", "无支撑特征"),
        ("no_fabricated", "无自造术语", "suspects", "可疑自造术语"),
        ("fluency", "流畅规范", None, None),
    ]
    for key, name, list_key, list_label in dims:
        d = result.get(key, {})
        out(f"\n【{name}】得分: {d.get('score', '?')}/5")
        out(f"  理由: {d.get('reason', '')}")
        if list_key:
            items = d.get(list_key, [])
            out(f"  {list_label}: {items if items else '无'}")

    ov = result.get("overall", {})
    out(f"\n【总体】得分: {ov.get('score', '?')}")
    out(f"  评价: {ov.get('summary', '')}")
    return "\n".join(lines)


def _read_triple(spec_p, abst_p, claim_p):
    """把三部分读出并拼成一整份文档文本，用于成对比较。"""
    spec = read_docx(spec_p)
    abst = read_docx(abst_p)
    claim = read_docx(claim_p)
    combined = f"【说明书】\n{spec}\n\n【摘要】\n{abst}\n\n【权利要求】\n{claim}"
    return spec, abst, claim, combined


def main():
    ap = argparse.ArgumentParser(description="LLM-as-a-Judge 专利质量评估")
    sub = ap.add_subparsers(dest="mode", required=True)

    # 子命令 score：绝对打分（原功能）
    sp = sub.add_parser("score", help="绝对打分（单份1-5分）[实测区分度不足,三阶段均4.5,仅作对照]")
    sp.add_argument("tech", help="交底书 docx")
    sp.add_argument("spec", help="说明书 docx")
    sp.add_argument("abst", help="摘要 docx")
    sp.add_argument("claim", help="权利要求 docx")
    sp.add_argument("--glossary", default="", help="术语表 txt（可选）")
    sp.add_argument("--tag", default="eval", help="结果文件标签，如 stage2_D1")

    # 子命令 pair：成对比较（新功能）
    pp = sub.add_parser("pair", help="成对比较（正反各问一次判胜负）[实测仍多判相当,灵敏度不足,仅作对照]")
    pp.add_argument("tech", help="交底书 docx")
    pp.add_argument("nameA", help="甲方标签，如 stage0")
    pp.add_argument("specA")
    pp.add_argument("abstA")
    pp.add_argument("claimA")
    pp.add_argument("nameB", help="乙方标签，如 stage2")
    pp.add_argument("specB")
    pp.add_argument("abstB")
    pp.add_argument("claimB")
    pp.add_argument("--glossary", default="")
    pp.add_argument("--out", default="", help="结果输出目录（默认当前目录）")

    # 子命令 extract：问题提取（逐条列举可计数问题）
    ep = sub.add_parser("extract", help="问题提取（逐条列举术语漂移/无支撑/自造术语并计数）[★推荐,阶段3主评估方法]")
    ep.add_argument("tech", help="交底书 docx")
    ep.add_argument("spec", help="说明书 docx")
    ep.add_argument("abst", help="摘要 docx")
    ep.add_argument("claim", help="权利要求 docx")
    ep.add_argument("--glossary", default="")
    ep.add_argument("--tag", default="extract", help="结果文件标签，如 stage2_D1")

    args = ap.parse_args()
    glossary = read_txt(args.glossary) if args.glossary and os.path.exists(args.glossary) else ""

    if args.mode == "score":
        tech = read_docx(args.tech)
        spec = read_docx(args.spec)
        abst = read_docx(args.abst)
        claim = read_docx(args.claim)
        result = evaluate(tech, spec, abst, claim, glossary)
        report = format_report(result, args.tag)
        print(report)
        outdir = os.path.dirname(os.path.abspath(args.spec))
        json_path = os.path.join(outdir, f"eval_result_{args.tag}.json")
        txt_path = os.path.join(outdir, f"eval_result_{args.tag}.txt")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n结果已写入:\n  {json_path}\n  {txt_path}")

    elif args.mode == "pair":
        tech = read_docx(args.tech)
        _, _, _, docA = _read_triple(args.specA, args.abstA, args.claimA)
        _, _, _, docB = _read_triple(args.specB, args.abstB, args.claimB)
        result = compare_pair(args.nameA, docA, args.nameB, docB, tech, glossary)
        report = format_pair_report(result)
        print(report)
        outdir = args.out if args.out else os.getcwd()
        tag = f"{args.nameA}_vs_{args.nameB}"
        json_path = os.path.join(outdir, f"pair_{tag}.json")
        txt_path = os.path.join(outdir, f"pair_{tag}.txt")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n结果已写入:\n  {json_path}\n  {txt_path}")

    elif args.mode == "extract":
        tech = read_docx(args.tech)
        spec = read_docx(args.spec)
        abst = read_docx(args.abst)
        claim = read_docx(args.claim)
        result = extract_issues(tech, spec, abst, claim, glossary)
        report = format_extract_report(result, args.tag)
        print(report)
        outdir = os.path.dirname(os.path.abspath(args.spec))
        json_path = os.path.join(outdir, f"extract_{args.tag}.json")
        txt_path = os.path.join(outdir, f"extract_{args.tag}.txt")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n结果已写入:\n  {json_path}\n  {txt_path}")


if __name__ == "__main__":
    main()
