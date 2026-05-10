# Official nuScenes Eval Diagram Prompt

## Primary Prompt (English)

Create a clean 16:9 infographic-style system diagram that explains an "Offline Official nuScenes Detection Evaluation" pipeline.

Visual style requirements:
- White solid background only (pure white, no gradients, no textures).
- Primary theme color must be **#ff9300** for key nodes, arrows, highlights, and titles.
- Secondary neutral colors allowed only for support elements: black text, light gray borders (#d9d9d9), very light gray panels (#f7f7f7).
- Flat modern vector style, high readability, minimal visual clutter.
- Keep strong contrast and spacious layout.

Content/layout requirements:
- Horizontal left-to-right flow with clear arrows:
  1) Input Predictions (`results_nusc.json`)
  2) GT + Metadata (`nuScenes val`)
  3) Official Filtering (`class_range`, points filter, bike-rack filter)
  4) Matching (`center distance thresholds: 0.5 / 1 / 2 / 4 m`)
  5) Metric Computation:
     - mAP (official definition)
     - TP Errors: `trans_err`, `scale_err`, `orient_err`, `vel_err`, `attr_err`
     - NDS aggregation (`mean_ap_weight = 5`)
  6) Final Outputs (`metrics_summary.json`, `metrics_details.json`, `NDS`)
- Add a small "Alignment" callout box:
  - Consistent with nuScenes devkit
  - Reproducible with MMDet3D evaluation
  - Comparable to leaderboard-style metrics
- Use simple icons (file, database, filter, target, chart, medal) in consistent style.
- Use concise labels and short captions only, all text in English.
- Ensure exact 16:9 composition.

Output constraints:
- No dark background.
- No photorealistic rendering.
- No watermark.
- No unrelated decorative elements.

---

## Optional Enhanced Prompt (English, more detailed)

Design a professional white-background 16:9 technical infographic titled "Offline Official nuScenes Evaluation". Use **#ff9300** as the dominant accent color for headings, flow arrows, and key metric blocks. Keep text black and structural separators light gray.

Show a six-step left-to-right pipeline:
1. Prediction input (`results_nusc.json`)
2. Ground-truth loading from nuScenes metadata
3. Official preprocessing/filtering (class range, valid points, bike-rack constraints)
4. Center-distance matching at multiple thresholds (0.5, 1.0, 2.0, 4.0 meters)
5. Metric engine (mAP + TP error metrics + official NDS formula)
6. Output artifacts and score summary

Include a compact "Why use official eval?" section with 4 short points:
- Standardized benchmark compliance
- Fair cross-model comparison
- Widely recognized reporting format
- Stable reproducibility

Use minimal vector icons, rounded rectangles, clean connectors, and generous whitespace. Maintain strict white background and 16:9 ratio.

