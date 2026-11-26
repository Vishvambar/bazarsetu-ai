# BazarSetu V3.0 — System 2 Evaluation Dataset

Organized collection of **43 unique craft products (52 images total)** covering single-view artisan categories, certification seals/price tags, ambiguous adversarial traps, and multi-view craft captures.

## Directory Structure

```text
dataset/
├── README.md
├── dataset_manifest.json          # Master index mapping product_id -> raw_views & golden_views
└── raw/                           # Unmodified input photographs
    ├── metalware/                 # 8 single-view products (metalware1..8)
    ├── pottery_wood/              # 8 single-view products (pottery_wood1..5,7,8,10)
    ├── handloom/                  # 8 single-view products (handloom1..8)
    ├── seals_tags/                # 8 products with seals/tags (seal1..5, tag1..3)
    ├── ambiguous/                 # 5 adversarial traps expecting needs_seller_review (ambiguous1..5)
    └── multiview/                 # 6 multi-view products (15 view images)
        ├── mv_handloom1/          # front.png, detail.png, back.png
        ├── mv_handloom2/          # front.png, detail.png
        ├── mv_metal1/             # front.png, detail.png
        ├── mv_pottery1/           # front.png, detail.png
        ├── mv_wood1/              # front.png, detail.png
        └── mv_wood2/              # front.png, detail.png, back.png, back_close.png
```

## Multi-View Naming Convention
* `front.png` → `front_view` (480px budget)
* `detail.png` → `detail_view` (macro craftsmanship / texture crop, 384px budget)
* `back.png` → `back_or_label_view` (reverse / label view, 384px budget)
* `back_close.png` → `escalation_crop_view` (tight high-res crop target for Step 6 escalation)

## Preparing Clean Stage G Cutouts (`storage/golden/`)
When generating System 1 Stage G white-background studio cutouts into `V3.0/storage/golden/`, run with `force=True` so smaller legacy benchmark images (`< 350px` side length: `metalware3`, `metalware4`, `pottery_wood2`, `pottery_wood3`) bypass Stage B quality advisory rejections cleanly.

