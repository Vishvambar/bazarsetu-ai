"""
BazarSetu V3.0 - Kaggle Profiler for System 2 (Gemma 4 12B + Gemma4ChatHandler).

Measures exact breakdown of:
  1. Installed Gemma4ChatHandler / MTMDBaseHandler signatures & VRAM placement
  2. Text-only decode speed (tok/s on cuda:1)
  3. Vision + plain JSON (no tool schema GBNF) speed
  4. Vision + tools/tool_choice (with higher max_tokens) speed & output truncation check
"""

from __future__ import annotations

import inspect
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.prepare_system2_golden import find_restore_archive, restore_golden_zip
from system2.engine import SYSTEM2_PROMPT, System2Config, System2Engine, prepare_view_data_uri


def print_vram(tag: str) -> None:
    if not torch.cuda.is_available():
        return
    parts = []
    for idx in range(torch.cuda.device_count()):
        free_b, total_b = torch.cuda.mem_get_info(idx)
        used_gb = (total_b - free_b) / (1024**3)
        total_gb = total_b / (1024**3)
        parts.append(f"cuda:{idx}={used_gb:.2f}/{total_gb:.2f}GB")
    print(f"[VRAM {tag}] " + " | ".join(parts), flush=True)


def main() -> int:
    golden_dir = PROJECT_ROOT / "storage" / "golden"
    archive = find_restore_archive(None)
    if archive:
        restore_golden_zip(archive, golden_dir)

    try:
        from llama_cpp.llama_multimodal import Gemma4ChatHandler, MTMDBaseHandler
        print("Gemma4ChatHandler.__init__:", inspect.signature(Gemma4ChatHandler.__init__))
        print("MTMDBaseHandler.__init__:", inspect.signature(MTMDBaseHandler.__init__))
        print("Gemma4ChatHandler.__call__:", inspect.signature(Gemma4ChatHandler.__call__))
    except Exception as exc:
        print("Handler inspect error:", exc)

    print_vram("before_load")
    cfg = System2Config(verbose=False, max_tokens=768)
    t0 = time.perf_counter()
    engine = System2Engine(config=cfg)
    print(f"[TIME load] {time.perf_counter() - t0:.2f}s", flush=True)
    print_vram("after_load")

    sample_img = golden_dir / "metalware1_front.png"
    if not sample_img.exists():
        print(f"Missing sample image: {sample_img}")
        return 1

    # 1. Pure text decode speed (Bypass chat_handler overhead vs Llama)
    t1 = time.perf_counter()
    txt_res = engine._llm.create_chat_completion(
        messages=[{"role": "user", "content": "List 20 Indian handicrafts separated by commas."}],
        max_tokens=60,
        temperature=0.1,
    )
    dt_txt = time.perf_counter() - t1
    txt_usage = txt_res.get("usage", {})
    comp_tok = txt_usage.get("completion_tokens", 0)
    print(
        f"[TEST A: Text-only] {dt_txt:.2f}s | prompt={txt_usage.get('prompt_tokens')} "
        f"| completion={comp_tok} ({comp_tok / max(dt_txt, 1e-6):.1f} tok/s)"
    )

    # 2. Image + Prompt WITHOUT tools/tool_choice (isolates mmproj + ViT + prompt prefill + decode)
    data_uri = prepare_view_data_uri(sample_img, max_dim=cfg.front_max_dim)
    t2 = time.perf_counter()
    img_res = engine._llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM2_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {
                        "type": "text",
                        "text": "Output one compact single-line JSON with keys domain_code, category, subcategory, primary_color, primary_material, craft_techniques.",
                    },
                ],
            },
        ],
        max_tokens=160,
        temperature=0.1,
    )
    dt_img = time.perf_counter() - t2
    img_usage = img_res.get("usage", {})
    print_vram("after_first_image")
    comp_img = img_usage.get("completion_tokens", 0)
    print(
        f"[TEST B: Image NO-tools] {dt_img:.2f}s | prompt={img_usage.get('prompt_tokens')} "
        f"| completion={comp_img} | finish={img_res['choices'][0].get('finish_reason')}"
    )
    print("  Sample B content:", repr(img_res["choices"][0]["message"].get("content", "")[:260]))

    # 3. Image + FULL tools + tool_choice with max_tokens=768
    messages, _ = engine.build_messages("metalware1", {"front_view": str(sample_img)})
    t3 = time.perf_counter()
    tool_res = engine._llm.create_chat_completion(
        messages=messages,
        tools=[engine._tool_def],
        tool_choice={"type": "function", "function": {"name": "submit_product_intelligence"}},
        max_tokens=768,
        temperature=0.1,
    )
    dt_tool = time.perf_counter() - t3
    t_usage = tool_res.get("usage", {})
    choice0 = tool_res["choices"][0]
    msg0 = choice0["message"]
    print(
        f"[TEST C: Image WITH-tools max_tokens=768] {dt_tool:.2f}s | prompt={t_usage.get('prompt_tokens')} "
        f"| completion={t_usage.get('completion_tokens')} | finish={choice0.get('finish_reason')}"
    )
    print("  tool_calls:", bool(msg0.get("tool_calls")), "content_len:", len(msg0.get("content") or ""))
    raw_dump = json.dumps(msg0, ensure_ascii=False)
    print("  raw head:", raw_dump[:350])
    print("  raw tail:", raw_dump[-350:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
