"""Run a SWE agent over SWE-bench Lite and write predictions.jsonl.

Usage (via WMI/bat):
    python eval_swe_inference.py \
        --model-path D:/zcy/models/Qwen3-14B-Instruct \
        --adapter-path outputs/swe_sft_model \
        --manifest D:/zcy/silr-swe-cache/swe-bench-lite.jsonl \
        --repo-cache D:/zcy/silr-swe-cache/repos \
        --output outputs/swe_eval/predictions-14B-sft.jsonl \
        --track 14B-sft

Produces JSONL lines {instance_id, model_name_or_path, model_patch} ready
for swebench.harness.run_evaluation.
"""
from __future__ import annotations

import argparse
import faulthandler
import gc
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from domains.swe.config import build_swe_domain_config
from domains.swe.manager import RepoManager
from domains.swe.scenarios import SWEBenchLoader

logger = logging.getLogger(__name__)


def setup_logging(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.with_suffix(".log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Crash to a dedicated file so native SIGSEGV (e.g. bnb on Blackwell)
    # leaves a trace instead of silent exit -1.
    fatal = output.with_suffix(".fatal.log")
    faulthandler.enable(file=open(fatal, "a", encoding="utf-8", buffering=1),
                        all_threads=True)


def load_model(model_path: str, adapter_path: str | None, quant: str = "bf16"):
    """Load Qwen3-14B / 32B.

    quant='bf16' (default): native bfloat16, ~28GB VRAM for 14B. This is the
        stable path on Blackwell — bnb 4-bit kernels crashed silently after 54
        instances (exit -1 no traceback, diagnosed as CUDA illegal memory
        access in bnb's sm_120+ path).
    quant='4bit': keep the old bitsandbytes 4-bit path for small-VRAM
        environments. Avoid on Blackwell.
    """
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if quant == "4bit":
        from transformers import BitsAndBytesConfig
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 over float16 on Blackwell
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=cfg, device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
            trust_remote_code=True,
        )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tok


def _load_done_ids(output: Path) -> set:
    """Resume support: collect instance_ids already written to the jsonl."""
    if not output.exists():
        return set()
    done = set()
    for line in output.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(json.loads(line)["instance_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


# Match `"locations": [ ... ]` for JSON-style tool call responses.
_LOCATIONS_ARRAY_RE = re.compile(r'["\']?locations["\']?\s*:\s*\[(.*?)\]', re.S)
_QUOTED_STRING_RE = re.compile(r'["\']((?:[^"\'\\]|\\.)*)["\']')
# Fallback: bare `path/to/file.py:123` — Qwen3 often ignores the tool schema
# and emits plain text lines like
#   :localize
#   astropy/modeling/separable.py:123
#   astropy/io/ascii/rst.py:45
_BARE_FILE_LINE_RE = re.compile(r"[\w./\-]+\.py:\d+")
_DIFF_BLOCK_RE = re.compile(r"(diff --git .*?)(?:```|\Z)", re.S)


def _extract_patch(text: str) -> str:
    """Extract a unified diff from LLM output, normalizing newlines.

    Handles three observed output formats:
    1. Tool-call JSON envelope `{"tool_name":"patch","params":{"patch":"diff --git ...\\n..."}}` —
       newlines arrive as literal `\\n` 2-char sequences. Must json.loads to
       unescape; otherwise git apply silently rejects every patch (the
       silent-corruption bug that collapsed all SFT predictions on first run).
    2. Markdown fenced block ` ```diff\\n...\\n``` ` — real newlines, regex captures cleanly.
    3. Bare diff in plain text — regex captures cleanly.
    """
    # Path 1: locate a JSON object containing a "patch" key whose value
    # starts with `diff --git`. Walk braces respecting strings/escapes.
    if '"patch"' in text:
        first_brace = text.find("{")
        if first_brace >= 0:
            depth = 0
            in_str = False
            esc = False
            end = -1
            for i in range(first_brace, len(text)):
                ch = text[i]
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end > first_brace:
                try:
                    obj = json.loads(text[first_brace : end + 1])
                except (json.JSONDecodeError, ValueError):
                    obj = None
                if obj is not None:
                    def _walk(o):
                        if isinstance(o, dict):
                            v = o.get("patch")
                            if isinstance(v, str) and "diff --git" in v:
                                return v
                            for vv in o.values():
                                r = _walk(vv)
                                if r:
                                    return r
                        elif isinstance(o, list):
                            for vv in o:
                                r = _walk(vv)
                                if r:
                                    return r
                        return None
                    p = _walk(obj)
                    if p:
                        return p.strip()
    # Path 2/3: legacy regex; if the captured text has literal `\n` but no
    # real newlines, the model wrote a JSON-escaped diff inline outside any
    # parseable envelope — unescape it.
    m = _DIFF_BLOCK_RE.search(text)
    if m:
        result = m.group(1).strip()
        if "\\n" in result and "\n" not in result:
            try:
                result = result.encode("utf-8", errors="replace").decode(
                    "unicode_escape"
                )
            except (UnicodeDecodeError, UnicodeEncodeError):
                pass
        return result
    return ""


def extract_locations(resp: str) -> list:
    """Robustly pull `locations` from an LLM response.

    1) Try JSON-style: `"locations": [ "file:line", ... ]`
    2) Fallback to bare `path/foo.py:123` tokens anywhere in the text.
    Returns [] only if neither pattern matches.
    """
    m = _LOCATIONS_ARRAY_RE.search(resp)
    if m:
        items = _QUOTED_STRING_RE.findall(m.group(1))
        if items:
            return items
    # Bare fallback — dedupe preserving order
    seen: set = set()
    out: list = []
    for loc in _BARE_FILE_LINE_RE.findall(resp):
        if loc not in seen:
            seen.add(loc)
            out.append(loc)
    return out


def read_code_snippets(
    work_dir: Path,
    locations: list,
    radius: int = 40,
    total_char_cap: int = 6000,
) -> str:
    """For each 'file:line', return ±radius lines with line numbers.

    Joined snippet text is capped at `total_char_cap`. Empty string if no
    location could be resolved. This is the Stage-2 context the LLM needs
    to produce non-hallucinated hunk headers (`@@ -N,M +N,M @@`) and real
    context lines — without it, zero-shot patches are cookie-cutter.
    """
    snippets: list[str] = []
    seen: set = set()
    for loc in locations:
        if not isinstance(loc, str) or ":" not in loc:
            continue
        try:
            file_part, line_part = loc.rsplit(":", 1)
            line_num = int(line_part.split("-")[0])  # accept "n-m" by taking start
        except (ValueError, TypeError):
            continue
        file_part = file_part.lstrip("/")
        file_path = work_dir / file_part
        if not file_path.is_file():
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = text.splitlines()
        start = max(0, line_num - 1 - radius)       # line_num is 1-indexed
        end = min(len(lines), line_num + radius)
        key = (file_part, start, end)
        if key in seen:
            continue
        seen.add(key)
        numbered = "\n".join(f"{i + 1:>5}: {lines[i]}" for i in range(start, end))
        snippets.append(f"=== {file_part} (lines {start + 1}-{end}) ===\n{numbered}")
    joined = "\n\n".join(snippets)
    if len(joined) > total_char_cap:
        joined = joined[:total_char_cap] + "\n... [truncated]"
    return joined


def load_few_shot_examples(path: Path, n: int) -> list[dict]:
    """Read first N records from a SWE-Gym JSONL and reshape as ICL turns.

    Each record yields a [user(problem), assistant(localize), user(patch_ack),
    assistant(patch)] sequence. Localize examples are synthesized from the
    gold patch's `--- a/file` headers — the model sees what good 2-stage
    behavior looks like on a known task before tackling the real one.
    """
    if not path.exists() or n <= 0:
        return []
    out: list[dict] = []
    seen = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if seen >= n or not line.strip():
            continue
        rec = json.loads(line)
        msgs = rec.get("messages", [])
        if not msgs:
            continue
        # Original SWE-Gym SFT records pack the task as user(json{instance,problem})
        # and assistant(json{tool_name:patch,params:{patch:...}}). Re-shape into
        # an explicit two-stage trajectory so the model learns the agent loop.
        user_obj = json.loads(msgs[1]["content"]) if len(msgs) > 1 else {}
        assist_obj = json.loads(msgs[2]["content"]) if len(msgs) > 2 else {}
        gold_patch = assist_obj.get("params", {}).get("patch", "")
        if not gold_patch:
            continue
        # Extract file paths from `--- a/<path>` headers as synthetic locations
        loc_files = re.findall(r"^--- a/(\S+)", gold_patch, flags=re.M)
        locations = [f"{p}:1" for p in loc_files[:5]] or ["?:1"]
        ex_user = json.dumps({
            "instance_id": user_obj.get("instance_id", "example"),
            "problem_statement": user_obj.get("problem_statement", ""),
        }, indent=2)
        ex_localize = json.dumps({
            "tool_name": "localize",
            "params": {"locations": locations},
        })
        ex_patch = json.dumps({
            "tool_name": "patch",
            "params": {"patch": gold_patch},
        })
        out.extend([
            {"role": "user", "content": ex_user},
            {"role": "user", "content": "Call the `localize` tool with candidate locations."},
            {"role": "assistant", "content": ex_localize},
            {"role": "user", "content": "Now call the `patch` tool with the full unified diff."},
            {"role": "assistant", "content": ex_patch},
        ])
        seen += 1
    return out


def run_one_instance(model, tok, cfg, mgr, few_shot_msgs: list[dict] | None = None) -> str:
    """Invoke the Agentless two-stage loop once. Return the final patch string."""
    obs = cfg.create_observer(mgr) if cfg.create_observer else None
    sys_prompt = cfg.build_system_prompt(mgr, cfg.build_tool_schemas(mgr))
    user_msg = json.dumps(obs.observe() if obs else {}, indent=2)

    # Stage 1: localize
    messages = [
        {"role": "system", "content": sys_prompt},
        *(few_shot_msgs or []),
        {"role": "user", "content": user_msg},
        {"role": "user", "content": "Call the `localize` tool with candidate locations."},
    ]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    # Decode only the newly-generated tokens (skip the prompt) so diagnostic
    # output reflects what the model actually produced.
    gen_tokens = out[0][inputs["input_ids"].shape[-1]:]
    localize_resp = tok.decode(gen_tokens, skip_special_tokens=True)
    logger.info(
        "[%s] stage1 raw first 400 chars: %r",
        mgr.instance.instance_id,
        localize_resp[:400],
    )
    locations = extract_locations(localize_resp)
    if not locations:
        locations = ["UNKNOWN:0"]
    loc_args = {"locations": locations}
    tools = cfg.create_toolset(mgr)
    tools["localize"].execute(**loc_args)

    # Read source snippets around each localized line. Without this the
    # model has no way to produce correct hunk offsets or context lines.
    snippet_text = read_code_snippets(Path(mgr.work_dir), loc_args.get("locations", []))
    logger.info(
        "[%s] stage1 locations=%s snippet_chars=%d work_dir=%s",
        mgr.instance.instance_id,
        loc_args.get("locations", [])[:3],
        len(snippet_text),
        str(mgr.work_dir)[:80],
    )

    # Stage 2: patch
    messages.append({"role": "assistant", "content": json.dumps(loc_args)})
    if snippet_text:
        stage2_prompt = (
            "Here are the localized code regions (line numbers on the left):\n\n"
            + snippet_text + "\n\n"
            "Now call the `patch` tool with a unified diff that fixes the bug.\n"
            "- Use the exact line numbers and text shown above so your hunk\n"
            "  header `@@ -<old_start>,<old_count> +<new_start>,<new_count> @@`\n"
            "  and context lines match the file byte-for-byte.\n"
            "- Keep the change minimal."
        )
    else:
        stage2_prompt = "Now call the `patch` tool with the full unified diff."
    messages.append({"role": "user", "content": stage2_prompt})

    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    gen_tokens = out[0][inputs["input_ids"].shape[-1]:]
    patch_resp = tok.decode(gen_tokens, skip_special_tokens=True)
    logger.info(
        "[%s] stage2 raw first 400 chars: %r",
        mgr.instance.instance_id,
        patch_resp[:400],
    )
    return _extract_patch(patch_resp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--repo-cache", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--track", required=True,
                    help="e.g. 14B-zs / 32B-zs / 14B-sft / 14B-sft-grpo")
    ap.add_argument("--limit", type=int, default=0, help="0 = all instances")
    ap.add_argument(
        "--quant", choices=["bf16", "4bit"], default="bf16",
        help="bf16 (default, stable on Blackwell) or 4bit (bnb; avoid on Blackwell)",
    )
    ap.add_argument(
        "--few-shot", type=int, default=0,
        help="Inject N synthetic 2-stage ICL turns from --few-shot-source.",
    )
    ap.add_argument(
        "--few-shot-source", default=None,
        help="JSONL of {messages:...} records (e.g. swe_sft.jsonl) to draw "
             "few-shot examples from. Required when --few-shot > 0.",
    )
    args = ap.parse_args()

    out = Path(args.output)
    setup_logging(out)
    logger.info(
        "starting inference track=%s model=%s adapter=%s quant=%s",
        args.track, args.model_path, args.adapter_path, args.quant,
    )

    model, tok = load_model(args.model_path, args.adapter_path, args.quant)
    loader = SWEBenchLoader(Path(args.manifest), Path(args.repo_cache))
    cfg = build_swe_domain_config(with_observer=True)

    few_shot_msgs: list[dict] = []
    if args.few_shot > 0:
        if not args.few_shot_source:
            raise SystemExit("--few-shot N requires --few-shot-source PATH")
        few_shot_msgs = load_few_shot_examples(Path(args.few_shot_source), args.few_shot)
        logger.info(
            "few-shot: %d turns injected (%d examples) from %s",
            len(few_shot_msgs), args.few_shot, args.few_shot_source,
        )

    ids = loader.list_instance_ids()
    if args.limit:
        ids = ids[: args.limit]

    # Resume: skip instance_ids that already have a record in the jsonl.
    done_ids = _load_done_ids(out)
    if done_ids:
        logger.info(
            "resume: %d already-processed instances in %s, will skip",
            len(done_ids), out,
        )
    remaining = [iid for iid in ids if iid not in done_ids]
    logger.info("%d instances to process (%d total)", len(remaining), len(ids))

    out.parent.mkdir(parents=True, exist_ok=True)
    # Append mode — preserves resume progress on every restart.
    with out.open("a", encoding="utf-8") as fh:
        for i, iid in enumerate(remaining):
            inst = None
            mgr = None
            try:
                inst = loader.load(iid)
                mgr = RepoManager(inst, work_dir=Path(inst.repo))  # reuse loader's work_root
                patch = run_one_instance(model, tok, cfg, mgr, few_shot_msgs)
            except Exception as e:
                logger.exception("instance %s failed: %s", iid, e)
                patch = ""
            finally:
                # Per-instance cleanup — Blackwell + bnb historically crashed
                # after 54 instances without this. Keeps VRAM and /tmp tidy.
                try:
                    if mgr is not None:
                        mgr.cleanup()
                    if inst is not None and inst.repo:
                        shutil.rmtree(inst.repo, ignore_errors=True)
                except Exception:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            rec = {
                "instance_id": iid,
                "model_name_or_path": args.track,
                "model_patch": patch,
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (i + 1) % 10 == 0:
                logger.info("progress %d/%d (global %d/%d)",
                            i + 1, len(remaining),
                            i + 1 + len(done_ids), len(ids))

    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
