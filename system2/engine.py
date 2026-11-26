"""
BazarSetu V3.0 - System 2 Inference Engine (GPU 1).

Serves quantized Gemma 4 12B Unified (gemma-4-12B-it-qat + mmproj-F16) strictly
pinned to cuda:1 via llama-cpp-python, leaving cuda:0 free for System 1 + SigLIP.
"""

from __future__ import annotations

import base64
import inspect
import io
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

from system2.schemas import ProductIntelligenceOutput

logger = logging.getLogger("BazarSetu.System2Engine")

PATCH_SIZE = 48  # Gemma 4 visual patch size (48x48 pixels)
VIEW_ORDER = ("front_view", "detail_view", "back_or_label_view", "escalation_crop_view")

SYSTEM2_PROMPT = """You are BazarSetu's System 2 Visual Catalog Intelligence Core for Indian handicrafts and ONDC retail.
Inspect the provided clean white-background studio product image(s) (front, optional detail, optional back/label) and emit a structured catalog record by calling the `submit_product_intelligence` tool.

STRICT RULES:
1. ONDC Domain Codes: Use "ONDC:RET16" for Pottery, Metalware, Woodcraft, Home Decor; "ONDC:RET12" for Sarees, Handloom Fabrics, Ethnic Apparel, Footwear; "ONDC:RET10" for Packaged Foods/Pickles.
2. Concise Bilingual Output (latency cap):
   - catalog_title_en: 40-70 chars.
   - catalog_title_hi: natural everyday Hindi in Devanagari.
   - description_en: 15-30 words.
   - description_hi: 15-25 words in simple Devanagari.
3. Anti-Hallucination & Seals:
   - ONLY list a seal in certification_seals ("Silk Mark", "Handloom Mark", "GI Tag", "FSSAI", "BIS Hallmark") if that exact logo or text is clearly visible in the images. Otherwise return [].
   - If visible packaging text or a seal is too small/blurry to read reliably, set text_legibility_confidence < 0.75 and needs_highres_label_crop = true.
4. Uncertainty & Seller Review:
   - If material or craft purity is visually ambiguous (e.g., brass vs gold-plated copper, handloom silk vs polyester blend, mixed multi-item bundle), set that material's confidence < 0.75 and add its dot-path (e.g., "materials[0].material_name") to audit_telemetry.needs_seller_review.
   - For unambiguous clear craft items, leave needs_seller_review empty [].
5. Craft techniques: use canonical snake_case tags when clearly visible: hand_hammered_mathar, lost_wax_cast, engraved, embossed, hand_painted, glazed, hand_carved, pit_fired, zari_brocade, embroidery, block_print, warli_painted."""


@dataclass
class System2Config:
    repo_id: str = os.getenv("SYSTEM2_MODEL_REPO", "unsloth/gemma-4-12B-it-qat-GGUF")
    model_filename: str = os.getenv("SYSTEM2_MODEL_FILE", "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf")
    mmproj_filename: str = os.getenv("SYSTEM2_MMPROJ_FILE", "mmproj-F16.gguf")
    local_weights_dir: Optional[str] = os.getenv("SYSTEM2_LOCAL_WEIGHTS_DIR")
    main_gpu: int = int(os.getenv("SYSTEM2_GPU_ID", "1"))
    tensor_split: List[float] = field(default_factory=lambda: [0.0, 1.0])
    n_gpu_layers: int = -1
    n_ctx: int = 4096
    n_threads: int = int(os.getenv("SYSTEM2_N_THREADS", "4"))
    temperature: float = 0.1
    max_tokens: int = 640
    front_max_dim: int = 480        # 10 * 48 patches
    secondary_max_dim: int = 384    # 8 * 48 patches
    escalation_max_dim: int = 672   # 14 * 48 patches
    enable_thinking: bool = False
    verbose: bool = False


def align_dim_to_patch(dim: int, max_dim: int) -> int:
    """Round a side length down/nearest to an exact multiple of 48 within [48, max_dim]."""
    clamped = min(dim, max_dim)
    aligned = (clamped // PATCH_SIZE) * PATCH_SIZE
    return max(PATCH_SIZE, aligned)


def compute_patch_aligned_size(width: int, height: int, max_dim: int) -> Tuple[int, int]:
    """Scale preserving aspect ratio, then snap both dimensions to multiples of 48."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")
    scale = min(1.0, max_dim / float(max(width, height)))
    scaled_w = max(PATCH_SIZE, int(round(width * scale)))
    scaled_h = max(PATCH_SIZE, int(round(height * scale)))
    target_w = max(PATCH_SIZE, min(max_dim, int(round(scaled_w / PATCH_SIZE)) * PATCH_SIZE))
    target_h = max(PATCH_SIZE, min(max_dim, int(round(scaled_h / PATCH_SIZE)) * PATCH_SIZE))
    return target_w, target_h


def prepare_view_data_uri(image_path: str | Path, max_dim: int) -> str:
    """Resize view cleanly to 48-pixel-aligned dimensions and encode as PNG/JPEG data URI."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"System 2 view image not found: {path}")
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        target_size = compute_patch_aligned_size(rgb.width, rgb.height, max_dim)
        if rgb.size != target_size:
            rgb = rgb.resize(target_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _resolve_json_schema_refs(node: Any, defs: Dict[str, Any]) -> Any:
    """
    Recursively inline JSON Schema $ref/$defs and flatten nullable anyOf so
    Gemma 4's tool declaration template sees every nested object's properties.
    """
    if isinstance(node, list):
        return [_resolve_json_schema_refs(item, defs) for item in node]
    if not isinstance(node, dict):
        return node

    if "$ref" in node:
        ref_key = node["$ref"].split("/")[-1]
        target = _resolve_json_schema_refs(defs.get(ref_key, {}), defs)
        merged = dict(target) if isinstance(target, dict) else {}
        for k, v in node.items():
            if k != "$ref":
                merged[k] = _resolve_json_schema_refs(v, defs)
        return merged

    if "anyOf" in node and isinstance(node["anyOf"], list):
        non_null = [
            opt for opt in node["anyOf"]
            if not (isinstance(opt, dict) and opt.get("type") == "null")
        ]
        if len(non_null) == 1:
            resolved = _resolve_json_schema_refs(non_null[0], defs)
            if isinstance(resolved, dict):
                merged = dict(resolved)
                for k, v in node.items():
                    if k != "anyOf":
                        merged[k] = _resolve_json_schema_refs(v, defs)
                return merged

    out: Dict[str, Any] = {}
    for k, v in node.items():
        if k == "$defs":
            continue
        out[k] = _resolve_json_schema_refs(v, defs)
    return out


def build_product_intelligence_tool() -> Dict[str, Any]:
    """Return inlined OpenAI tool spec with zero $ref/$defs for Gemma4ChatHandler."""
    raw_schema = ProductIntelligenceOutput.model_json_schema()
    defs = raw_schema.get("$defs", {})
    inlined = _resolve_json_schema_refs(raw_schema, defs)
    inlined.pop("$defs", None)
    return {
        "type": "function",
        "function": {
            "name": "submit_product_intelligence",
            "description": "Submit the extracted ONDC product intelligence JSON record.",
            "parameters": inlined,
        },
    }


def _strip_thought_and_fences(raw_text: str) -> str:
    """Remove any empty/residual Gemma 4 thought blocks and markdown code fences."""
    cleaned = re.sub(r"<\|?channel>thought.*?<channel\|>", "", raw_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<\|?thought\|?>.*?<\|?thought\|?>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<\|?thought\|?>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


def extract_tool_or_json_payload(response: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and parse JSON dict from forced tool call, falling back to message content."""
    choices = response.get("choices") or []
    if not choices:
        raise ValueError(f"Model returned empty choices: {response}")
    msg = choices[0].get("message") or {}

    tool_calls = msg.get("tool_calls") or []
    if tool_calls:
        fn_block = tool_calls[0].get("function") or {}
        args = fn_block.get("arguments")
        if isinstance(args, dict):
            return args
        if isinstance(args, str) and args.strip():
            return json.loads(_strip_thought_and_fences(args))

    content = msg.get("content") or ""
    cleaned = _strip_thought_and_fences(content)
    if not cleaned:
        raise ValueError("Model returned neither tool_calls arguments nor JSON content.")
    # Locate outer JSON object if extra tokens surround it
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


class System2Engine:
    """
    Thread-safe Gemma 4 12B QAT serving engine isolated onto GPU 1 (cuda:1).
    Accepts an optional pre-injected completion callable for offline unit testing.
    """

    def __init__(
        self,
        config: Optional[System2Config] = None,
        lazy_load: bool = False,
        completion_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    ):
        self.cfg = config or System2Config()
        self._lock = threading.Lock()
        self._llm: Any = None
        self._chat_handler: Any = None
        self._completion_fn = completion_fn
        self._tool_def = build_product_intelligence_tool()
        self._base_dir = Path(__file__).resolve().parent.parent

        if not lazy_load and self._completion_fn is None:
            self.load()

    def _resolve_model_paths(self) -> Tuple[str, str]:
        """Resolve local paths (or download via huggingface_hub) for QAT GGUF & mmproj."""
        if self.cfg.local_weights_dir:
            wdir = Path(self.cfg.local_weights_dir)
            model_path = wdir / self.cfg.model_filename
            mmproj_path = wdir / self.cfg.mmproj_filename
            if model_path.exists() and mmproj_path.exists():
                return str(model_path), str(mmproj_path)

        from huggingface_hub import hf_hub_download

        logger.info("Downloading/resolving %s (%s)...", self.cfg.repo_id, self.cfg.model_filename)
        model_path = hf_hub_download(repo_id=self.cfg.repo_id, filename=self.cfg.model_filename)
        mmproj_path = hf_hub_download(repo_id=self.cfg.repo_id, filename=self.cfg.mmproj_filename)
        return model_path, mmproj_path

    @staticmethod
    def ensure_ggml_backends_loaded() -> None:
        """
        Load dynamic ggml backend plugins (e.g., libggml-cuda.so in llama-cpp-python >= 0.4)
        before checking llama_supports_gpu_offload() or initializing mmproj.
        """
        import ctypes
        import llama_cpp

        try:
            from llama_cpp._ggml import ggml_backend_load_all_from_path
        except ImportError:
            return
        llama_cpp.llama_backend_init()
        lib_dir = Path(llama_cpp.__file__).resolve().parent / "lib"
        if lib_dir.exists():
            ggml_backend_load_all_from_path(ctypes.c_char_p(str(lib_dir).encode("utf-8")))

    def load(self) -> None:
        if self._llm is not None or self._completion_fn is not None:
            return
        with self._lock:
            if self._llm is not None:
                return
            import llama_cpp
            from llama_cpp import Llama
            try:
                from llama_cpp.llama_multimodal import Gemma4ChatHandler
            except ImportError:
                from llama_cpp.llama_chat_format import Gemma4ChatHandler

            self.ensure_ggml_backends_loaded()
            if not llama_cpp.llama_supports_gpu_offload():
                raise RuntimeError(
                    "llama-cpp-python was compiled WITHOUT CUDA GPU offload! "
                    "Re-run kaggle/setup_kaggle.sh to install the prebuilt CUDA wheel."
                )

            model_path, mmproj_path = self._resolve_model_paths()

            # Inspect Gemma4ChatHandler signature cleanly across llama-cpp-python versions
            handler_sig = inspect.signature(Gemma4ChatHandler.__init__).parameters
            handler_kwargs: Dict[str, Any] = {"verbose": self.cfg.verbose}
            if "mmproj_path" in handler_sig:
                handler_kwargs["mmproj_path"] = mmproj_path
            elif "clip_model_path" in handler_sig:
                handler_kwargs["clip_model_path"] = mmproj_path
            if "enable_thinking" in handler_sig:
                handler_kwargs["enable_thinking"] = self.cfg.enable_thinking
            if "flash_attn" in handler_sig:
                handler_kwargs["flash_attn"] = False
            if "use_gpu" in handler_sig:
                handler_kwargs["use_gpu"] = True

            self._chat_handler = Gemma4ChatHandler(**handler_kwargs)

            llama_kwargs: Dict[str, Any] = {
                "model_path": model_path,
                "chat_handler": self._chat_handler,
                "n_ctx": self.cfg.n_ctx,
                "n_threads": self.cfg.n_threads,
                "n_gpu_layers": self.cfg.n_gpu_layers,
                "main_gpu": self.cfg.main_gpu,
                "tensor_split": self.cfg.tensor_split,
                "verbose": self.cfg.verbose,
            }
            logger.info(
                "Loading Gemma 4 12B onto cuda:%d (tensor_split=%s, enable_thinking=%s)...",
                self.cfg.main_gpu,
                self.cfg.tensor_split,
                self.cfg.enable_thinking,
            )
            self._llm = Llama(**llama_kwargs)
            self.log_gpu_isolation()

    @staticmethod
    def log_gpu_isolation() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                for idx in range(torch.cuda.device_count()):
                    free_b, total_b = torch.cuda.mem_get_info(idx)
                    used_gb = (total_b - free_b) / (1024**3)
                    total_gb = total_b / (1024**3)
                    logger.info("VRAM status cuda:%d -> %.2f / %.2f GB used", idx, used_gb, total_gb)
        except Exception as exc:
            logger.debug("Could not inspect CUDA VRAM allocation: %s", exc)

    def build_messages(
        self,
        product_id: str,
        views: Dict[str, str],
        escalation_mode: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Build 48-aligned multi-view chat completion messages cleanly."""
        content_parts: List[Dict[str, Any]] = []
        views_used: List[str] = []

        ordered_keys = [k for k in VIEW_ORDER if k in views] + [
            k for k in views if k not in VIEW_ORDER
        ]
        for vkey in ordered_keys:
            raw_path = Path(views[vkey])
            resolved_path = raw_path if raw_path.is_absolute() else (self._base_dir / raw_path)
            if escalation_mode or vkey == "escalation_crop_view":
                max_dim = self.cfg.escalation_max_dim
            elif vkey == "front_view":
                max_dim = self.cfg.front_max_dim
            else:
                max_dim = self.cfg.secondary_max_dim

            data_uri = prepare_view_data_uri(resolved_path, max_dim=max_dim)
            content_parts.append({"type": "text", "text": f"[VIEW: {vkey}]"})
            content_parts.append({"type": "image_url", "image_url": {"url": data_uri}})
            views_used.append(vkey)

        if not views_used:
            raise ValueError(f"No views provided for product_id={product_id}")

        content_parts.append(
            {
                "type": "text",
                "text": (
                    f"Analyze product_id='{product_id}' across views {views_used}. "
                    "Set audit_telemetry.views_processed accurately and call `submit_product_intelligence`."
                ),
            }
        )

        messages = [
            {"role": "system", "content": SYSTEM2_PROMPT},
            {"role": "user", "content": content_parts},
        ]
        return messages, views_used

    def predict(
        self,
        product_id: str,
        views: Dict[str, str],
        escalation_mode: bool = False,
    ) -> Dict[str, Any]:
        """Thread-safe prediction returning a validated ProductIntelligenceOutput dict."""
        if self._llm is None and self._completion_fn is None:
            self.load()

        messages, views_used = self.build_messages(product_id, views, escalation_mode=escalation_mode)

        with self._lock:
            fn = self._completion_fn or self._llm.create_chat_completion
            kwargs: Dict[str, Any] = {
                "messages": messages,
                "tools": [self._tool_def],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "submit_product_intelligence"},
                },
                "temperature": self.cfg.temperature,
                "max_tokens": self.cfg.max_tokens,
            }
            sig = inspect.signature(fn).parameters
            if "chat_template_kwargs" in sig:
                kwargs["chat_template_kwargs"] = {"enable_thinking": self.cfg.enable_thinking}

            response = fn(**kwargs)

        payload = extract_tool_or_json_payload(response)
        payload["product_id"] = product_id
        telemetry = payload.setdefault("audit_telemetry", {})
        if not telemetry.get("views_processed"):
            telemetry["views_processed"] = views_used

        validated = ProductIntelligenceOutput.model_validate(payload)
        return validated.model_dump(mode="json")

