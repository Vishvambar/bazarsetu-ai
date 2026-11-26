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

from system2.schemas import (
    CompositionType,
    ONDCDomainCode,
    PatternCategory,
    PhysicalScale,
    ProductIntelligenceOutput,
    SurfaceFinish,
)

logger = logging.getLogger("BazarSetu.System2Engine")

PATCH_SIZE = 48  # Gemma 4 visual patch size (48x48 pixels)
VIEW_ORDER = ("front_view", "detail_view", "back_or_label_view", "escalation_crop_view")

SYSTEM2_PROMPT = """You are BazarSetu's System 2 Visual Catalog Intelligence Core for Indian handicrafts and ONDC retail.
Inspect ALL views carefully and output ONLY ONE single-line compact JSON object in English (no markdown fences, no commentary, omit empty optional lists). Keep "desc" 8-14 words and "kw" <= 2 short tags.

Exact Enum & Visual Classification Rules:
- domain: "ONDC:RET16" (Pottery, Metalware, Woodcraft, Stone/Marble, Home Decor, decorative Jewellery Boxes) | "ONDC:RET12" (Sarees, Fabrics, Kurtas/Dhotis/Apparel, Footwear) | "ONDC:RET10" (Packaged Foods/Pickles jars)
- comp: "single_item" (1 garment/saree/object even with hangtags) | "matched_set" (2+ distinct/coordinated bowls, musicians/figurines, kurta+dupatta, 2 fabric bolts) | "multipack" (2+ identical/near-identical pieces e.g. matching oil diyas, identical bud vases) | "product_with_packaging" (labelled jar OR footwear/product photographed with packaging box)
- scale: "pocket_wearable" (footwear/chappals, jewelry) | "handheld" (mugs, cups, tumblers, small single bowls, small oil diyas) | "tabletop" (pickle jars, vases, teapots, kalash, urlis, 2-bowl sets, carved bowls, pillar candle stands, boxes, figurines, trays) | "floor_standing" (ONLY >=3ft tall floor samai lamps or large planters) | "draped_fabric" (ALL sarees, dhotis, veshtis, kurtas, kaftans, dupattas, fabric lengths)
- color_fam: EXACT one of ["Red","Blue","Green","Yellow","Orange","Pink","Purple","Grey","Black","White","Brown","Gold","Silver","Beige","Multicolor"] (Gold=brass/bronze/gold; Silver=pewter/silver/white-metal tumbler; Orange=burnished copper/saffron; Brown=wood/terracotta clay/leather/amber pickle jars; Black=dark oxidized black metal; White=white marble; Purple=deep violet-purple)
- finish: "matte"|"glossy"|"metallic"|"oxidized_antique"|"raw_natural"|"woven_textile"|"hammered_mathar"|"glazed"
- mat_fam: "Metal"|"Textile"|"Wood"|"Clay_Ceramic"|"Leather"|"Stone"|"Glass"|"Natural_Fiber"|"Paper_Bamboo" (carved/painted marble stand=Stone; decorative ornate jewellery box base=Wood)
- pat: subset of ["solid","geometric","floral","traditional_motif","warli_tribal","striped","checked","abstract"]
- craft: emit ALL applicable visible tags from ["hand_hammered_mathar","lost_wax_cast","engraved","embossed","hand_painted","glazed","hand_carved","pit_fired","zari_brocade","embroidery","block_print","warli_painted"] (only use hand_hammered_mathar when round dimples cover metal; sculpted bronze/brass figurines or flasks=lost_wax_cast; etched lines on tumblers/vases/terracotta=engraved; raised 3D relief on teapot/studded box=embossed; shiny painted ceramic=BOTH glazed+hand_painted; rustic matte terracotta=pit_fired; white tribal stick-figures on wooden tray=warli_painted; woven gold/metallic saree=zari_brocade)
- seals: inspect EVERY paper hangtag, cloth tag, AND jar label from top to bottom (a single vertical white card next to a jar often stacks 3-4 DIFFERENT seals above 'India Handloom' — do NOT skip the card!). Extract ALL matching seals from ["Silk Mark","Handloom Mark","GI Tag","FSSAI","BIS Hallmark"] (red butterfly/'SILK MARK'="Silk Mark"; circular wheel/hand 'HANDLOOM MARK'="Handloom Mark"; teardrop pin/'GI Tag'="GI Tag"; jar 'fssai'/green veg dot="FSSAI"; omit only 'India Handloom'/'India Handmade' itself). Else []
- text: true if tag/label text visible, else false; legib: 0.0-1.0; crop: true ONLY if blurry
- review: set "mat_conf": 0.65 and "review": ["materials[0].material_name"] ONLY for these cases: (a) 1 vessel fusing ceramic+cane OR 2 unlabelled contrasting fabric bolts side-by-side; (b) off-white/beige kurta+dupatta flat-lay beside brass bangles/bowl props where fiber purity is unlabelled; (c) smooth mirror-polished plain brass bowl with no carvings/dimples OR paired engraved bowls showing mismatched copper-toned rim vs aged brass purity. Keep clear pottery/wood/carved/hammered/zari/red-or-yellow garments/tagged items as "review": []!

Example schema (fill values for the actual input image):
{"product":"<name>","title":"<40-70 char EN title>","desc":"<8-14 word EN sentence>","comp":"single_item","scale":"tabletop","domain":"ONDC:RET16","cat":"<category>","subcat":"<subcategory>","color":"<color>","color_fam":"<color_fam>","finish":"matte","mat":"<material>","mat_fam":"Clay_Ceramic","mat_conf":0.95,"craft":[],"seals":[],"text":false,"legib":1.0,"crop":false,"conf":0.94,"review":[],"kw":["<kw1>","<kw2>"]}"""

LEGACY_TOOLS_PROMPT = """You are BazarSetu's System 2 Visual Catalog Intelligence Core for Indian handicrafts and ONDC retail.
Inspect the provided clean white-background studio product image(s) and emit an English-only structured catalog record by calling `submit_product_intelligence`."""


@dataclass
class System2Config:
    repo_id: str = os.getenv("SYSTEM2_MODEL_REPO", "unsloth/gemma-4-12B-it-qat-GGUF")
    model_filename: str = os.getenv("SYSTEM2_MODEL_FILE", "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf")
    mmproj_filename: str = os.getenv("SYSTEM2_MMPROJ_FILE", "mmproj-F16.gguf")
    local_weights_dir: Optional[str] = os.getenv("SYSTEM2_LOCAL_WEIGHTS_DIR")
    main_gpu: int = int(os.getenv("SYSTEM2_GPU_ID", "1"))
    tensor_split: List[float] = field(default_factory=lambda: [0.0, 1.0])
    n_gpu_layers: int = -1
    n_ctx: int = 3072
    n_batch: int = 1024
    n_threads: int = int(os.getenv("SYSTEM2_N_THREADS", "4"))
    temperature: float = 0.1
    max_tokens: int = 380
    front_max_dim: int = 768        # 16 * 48 patches (keeps 16:9 multi-seal cards crisp)
    secondary_max_dim: int = 384    # 8 * 48 patches
    escalation_max_dim: int = 864   # 18 * 48 patches (back_or_label_view / escalations)
    enable_thinking: bool = False
    use_tools: bool = False         # Disabled by default: tools GBNF drops T4 speed 2.1x on 262k vocab
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
        if k in ("$defs", "title", "description", "default"):
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


def _loads_relaxed_json(text: str) -> Dict[str, Any]:
    """Parse JSON after stripping minor trailing commas if needed."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = re.sub(r",(\s*[}\]])", r"\1", text)
        return json.loads(repaired)


def _coerce_domain(raw_val: Any) -> str:
    val = str(raw_val or "ONDC:RET16").strip().upper()
    if val.startswith("RET"):
        val = f"ONDC:{val}"
    valid = {e.value for e in ONDCDomainCode}
    return val if val in valid else ONDCDomainCode.HOME_DECOR.value


def _coerce_enum(raw_val: Any, enum_cls: Any, aliases: Dict[str, str], default: str) -> str:
    val = str(raw_val or default).strip().lower().replace("-", "_").replace(" ", "_")
    val = aliases.get(val, val)
    valid = {e.value for e in enum_cls}
    return val if val in valid else default


def _coerce_color_family(raw_val: Any) -> str:
    val = str(raw_val or "Brown").strip()
    low = val.lower()
    alias_map = {
        "copper": "Orange",
        "terracotta": "Brown",
        "rust": "Brown",
        "brass": "Gold",
        "bronze": "Gold",
        "golden": "Gold",
        "pewter": "Silver",
        "steel": "Silver",
        "metallic": "Silver",
        "ivory": "White",
        "cream": "Beige",
        "off_white": "White",
        "off-white": "White",
        "charcoal": "Black",
        "gray": "Grey",
        "multi": "Multicolor",
        "multicolour": "Multicolor",
    }
    if low in alias_map:
        return alias_map[low]
    return val.title()


def _clamp_conf(raw_val: Any, default: float = 0.9) -> float:
    try:
        v = float(raw_val)
    except (TypeError, ValueError):
        v = default
    return max(0.0, min(1.0, v))


def normalize_wire_payload(
    raw: Dict[str, Any],
    product_id: str,
    views_used: List[str],
) -> Dict[str, Any]:
    """
    Normalize either full ProductIntelligenceOutput dicts or compact flat wire JSON
    (used to keep Gemma 4 completion tokens <= 180 and eliminate schema GBNF latency)
    into the locked ProductIntelligenceOutput schema with defensive enum coercion.
    """
    if "product_identity" in raw and "ondc_taxonomy" in raw:
        out = dict(raw)
        out["product_id"] = product_id
        telemetry = dict(out.get("audit_telemetry") or {})
        if not telemetry.get("views_processed"):
            telemetry["views_processed"] = views_used
        out["audit_telemetry"] = telemetry
        return out

    subcat = str(raw.get("subcat") or raw.get("product") or "Handcrafted Item")
    conf_default = _clamp_conf(raw.get("conf"), 0.9)

    raw_mats = raw.get("mats")
    materials: List[Dict[str, Any]] = []
    if isinstance(raw_mats, list) and raw_mats:
        for item in raw_mats:
            if not isinstance(item, dict):
                continue
            materials.append(
                {
                    "material_name": item.get("material_name") or item.get("name") or item.get("mat") or "Unknown",
                    "material_family": item.get("material_family") or item.get("fam") or item.get("mat_fam") or "Metal",
                    "location": item.get("location") or item.get("loc") or "main_body",
                    "confidence": _clamp_conf(item.get("confidence", item.get("conf", conf_default)), conf_default),
                }
            )
    if not materials:
        materials.append(
            {
                "material_name": str(raw.get("mat") or "Handcrafted Material"),
                "material_family": str(raw.get("mat_fam") or "Metal"),
                "location": str(raw.get("mat_loc") or "main_body"),
                "confidence": _clamp_conf(raw.get("mat_conf", conf_default), conf_default),
            }
        )

    sec_colors: List[Dict[str, str]] = []
    for c in (raw.get("sec_colors") or [])[:2]:
        if isinstance(c, dict) and c.get("name") and c.get("standard_family"):
            sec_colors.append(
                {"name": str(c["name"]), "standard_family": _coerce_color_family(c["standard_family"])}
            )
        elif isinstance(c, dict) and c.get("c") and c.get("f"):
            sec_colors.append({"name": str(c["c"]), "standard_family": _coerce_color_family(c["f"])})

    pat_aliases = {
        "floral_motif": "floral",
        "tribal": "warli_tribal",
        "warli": "warli_tribal",
        "stripes": "striped",
        "checkered": "checked",
        "checks": "checked",
        "plain": "solid",
        "none": "solid",
        "traditional": "traditional_motif",
        "ethnic": "traditional_motif",
    }
    valid_pats = {e.value for e in PatternCategory}
    pat_cats: List[str] = []
    raw_pat = raw.get("pat") or []
    if isinstance(raw_pat, str):
        raw_pat = [raw_pat]
    for p in raw_pat:
        norm_p = pat_aliases.get(str(p).strip().lower().replace("-", "_").replace(" ", "_"), str(p).strip().lower().replace("-", "_").replace(" ", "_"))
        if norm_p in valid_pats and norm_p not in pat_cats:
            pat_cats.append(norm_p)

    seals = [str(s).strip() for s in (raw.get("seals") or []) if s and str(s).strip().lower() != "none"]
    snippets = [str(s).strip() for s in (raw.get("snippets") or [])[:3] if s]
    has_text = bool(raw.get("text", bool(seals or snippets or raw.get("brand"))))

    raw_review = raw.get("review")
    if isinstance(raw_review, bool):
        review_list = ["materials[0].material_name"] if raw_review else []
    elif isinstance(raw_review, str):
        review_list = [raw_review] if raw_review.strip() else []
    elif isinstance(raw_review, list):
        review_list = [str(x) for x in raw_review if x]
    else:
        review_list = []

    comp_val = _coerce_enum(
        raw.get("comp"),
        CompositionType,
        {
            "single": "single_item",
            "set": "matched_set",
            "pair": "matched_set",
            "combo": "multipack",
            "pack": "multipack",
            "packaged": "product_with_packaging",
        },
        CompositionType.SINGLE_ITEM.value,
    )
    scale_val = _coerce_enum(
        raw.get("scale"),
        PhysicalScale,
        {
            "pocket": "pocket_wearable",
            "wearable": "pocket_wearable",
            "small": "handheld",
            "medium": "tabletop",
            "large": "floor_standing",
            "draped": "draped_fabric",
            "fabric": "draped_fabric",
            "apparel": "draped_fabric",
        },
        PhysicalScale.TABLETOP.value,
    )
    finish_val = _coerce_enum(
        raw.get("finish"),
        SurfaceFinish,
        {
            "polished": "glossy",
            "shiny": "glossy",
            "oxidized": "oxidized_antique",
            "antique": "oxidized_antique",
            "aged": "oxidized_antique",
            "hammered": "hammered_mathar",
            "mathar": "hammered_mathar",
            "hand_hammered": "hammered_mathar",
            "woven": "woven_textile",
            "fabric": "woven_textile",
            "textile": "woven_textile",
            "natural": "raw_natural",
            "raw": "raw_natural",
            "terracotta": "raw_natural",
            "rustic": "raw_natural",
        },
        SurfaceFinish.MATTE.value,
    )

    return {
        "product_id": product_id,
        "product_identity": {
            "apparent_product": str(raw.get("product") or subcat),
            "catalog_title_en": str(raw.get("title") or f"Handcrafted {subcat}"),
            "description_en": str(raw.get("desc") or f"Handcrafted {subcat.lower()} artisan product."),
            "composition_type": comp_val,
            "set_contents": [str(x) for x in (raw.get("set") or [])],
            "physical_scale": scale_val,
        },
        "ondc_taxonomy": {
            "domain_code": _coerce_domain(raw.get("domain")),
            "category": str(raw.get("cat") or "Home Decor"),
            "subcategory": subcat,
            "generic_name": str(raw.get("gen") or subcat),
            "confidence": _clamp_conf(raw.get("dom_conf", conf_default), conf_default),
        },
        "color_profile": {
            "primary_color": {
                "name": str(raw.get("color") or raw.get("color_fam") or "Neutral"),
                "standard_family": _coerce_color_family(raw.get("color_fam")),
            },
            "secondary_colors": sec_colors,
            "surface_finish": finish_val,
            "color_distribution": str(raw.get("dist") or "solid"),
        },
        "materials": materials,
        "pattern_and_craft": {
            "has_pattern": bool(raw.get("has_pat", bool(pat_cats and pat_cats != ["solid"]))),
            "pattern_categories": pat_cats,
            "craft_techniques": [str(c).strip().lower() for c in (raw.get("craft") or []) if c],
            "motif_description": raw.get("motif"),
        },
        "structural_components": list(raw.get("parts") or []),
        "text_and_seals": {
            "has_visible_text": has_text,
            "brand_or_artisan_mark": raw.get("brand"),
            "certification_seals": seals,
            "visible_text_snippets": snippets,
            "text_legibility_confidence": _clamp_conf(raw.get("legib", 1.0), 1.0),
            "needs_highres_label_crop": bool(raw.get("crop", False)),
        },
        "marketplace_enrichment": {
            "target_audience": str(raw.get("aud") or "home_utility"),
            "occasions": [str(x) for x in (raw.get("occ") or [])[:3]],
            "search_keywords": [str(x) for x in (raw.get("kw") or [])[:6]],
        },
        "audit_telemetry": {
            "views_processed": views_used,
            "overall_confidence": conf_default,
            "needs_seller_review": review_list,
        },
    }


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
            return _loads_relaxed_json(_strip_thought_and_fences(args))

    content = msg.get("content") or ""
    cleaned = _strip_thought_and_fences(content)
    if not cleaned:
        raise ValueError("Model returned neither tool_calls arguments nor JSON content.")
    # Locate outer JSON object if extra tokens surround it
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    return _loads_relaxed_json(cleaned)


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

            # Inspect Gemma4ChatHandler signature (supports **kwargs delegating to MTMDBaseHandler)
            handler_sig = inspect.signature(Gemma4ChatHandler.__init__).parameters
            has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in handler_sig.values())
            handler_kwargs: Dict[str, Any] = {"verbose": self.cfg.verbose}
            if "mmproj_path" in handler_sig or has_varkw:
                handler_kwargs["mmproj_path"] = mmproj_path
            elif "clip_model_path" in handler_sig:
                handler_kwargs["clip_model_path"] = mmproj_path
            if "enable_thinking" in handler_sig or has_varkw:
                handler_kwargs["enable_thinking"] = self.cfg.enable_thinking
            if "flash_attn" in handler_sig or has_varkw:
                handler_kwargs["flash_attn"] = False
            if "use_gpu" in handler_sig or has_varkw:
                handler_kwargs["use_gpu"] = True

            self._chat_handler = Gemma4ChatHandler(**handler_kwargs)

            llama_kwargs: Dict[str, Any] = {
                "model_path": model_path,
                "chat_handler": self._chat_handler,
                "n_ctx": self.cfg.n_ctx,
                "n_batch": self.cfg.n_batch,
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
            if escalation_mode or vkey in ("escalation_crop_view", "back_or_label_view"):
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

        sys_prompt = LEGACY_TOOLS_PROMPT if self.cfg.use_tools else SYSTEM2_PROMPT
        tail_instruction = (
            f"Analyze product_id='{product_id}' across views {views_used}. "
            "Call `submit_product_intelligence`."
            if self.cfg.use_tools
            else f"Analyze views {views_used} for product_id='{product_id}' and output ONLY the single-line compact JSON object."
        )
        content_parts.append({"type": "text", "text": tail_instruction})

        messages = [
            {"role": "system", "content": sys_prompt},
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

        last_err: Optional[Exception] = None
        for attempt in range(2):
            attempt_messages = messages
            if attempt == 1 and last_err is not None:
                attempt_messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            f"Previous JSON failed ({last_err}). "
                            "Reply with ONLY one valid single-line JSON object matching all keys."
                        ),
                    }
                ]

            with self._lock:
                fn = self._completion_fn or self._llm.create_chat_completion
                kwargs: Dict[str, Any] = {
                    "messages": attempt_messages,
                    "temperature": 0.0 if attempt == 1 else self.cfg.temperature,
                    "max_tokens": 768 if self.cfg.use_tools else self.cfg.max_tokens,
                }
                if self.cfg.use_tools:
                    kwargs["tools"] = [self._tool_def]
                    kwargs["tool_choice"] = {
                        "type": "function",
                        "function": {"name": "submit_product_intelligence"},
                    }
                sig = inspect.signature(fn).parameters
                if "chat_template_kwargs" in sig:
                    kwargs["chat_template_kwargs"] = {"enable_thinking": self.cfg.enable_thinking}

                response = fn(**kwargs)

            try:
                payload = extract_tool_or_json_payload(response)
                normalized = normalize_wire_payload(payload, product_id=product_id, views_used=views_used)
                validated = ProductIntelligenceOutput.model_validate(normalized)
                return validated.model_dump(mode="json")
            except Exception as exc:
                last_err = exc
                logger.warning("Attempt %d validation failed for %s: %s", attempt + 1, product_id, exc)

        raise RuntimeError(f"System2Engine failed for {product_id} after 2 attempts: {last_err}")

