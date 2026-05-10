# Safety-Critical Eval Diagram Prompt

## Primary Prompt (English)

Create a clean 16:9 infographic-style system diagram that explains a "Safety-Critical nuScenes Detection Evaluation" pipeline.

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
  3) Distance Filter (`class_range + safety_max_dist`)
  4) Matching (`center distance thresholds`)
  5) Metric Computation:
     - Weighted mAP (`class_ap_weights`)
     - Scale Error (`w/l/h weighted`)
     - Orientation Error (`yaw-difference piecewise weighting`)
     - Velocity Error (`yaw-difference piecewise weighting`)
  6) Final Outputs (`Safety-Critical NDS`, `metrics_summary.json`)
- Add a small comparison callout box on the right: "Official Eval vs Safety-Critical Eval" with 3 short bullets:
  - Near-range prioritization
  - Class-aware weighting
  - Risk-aware orientation/velocity weighting
- Use simple icons (file, filter, target, chart, shield) in matching style.
- Include concise labels and short captions only, all text in English.
- Ensure the final composition is exactly 16:9.

Output constraints:
- No dark background.
- No photorealistic rendering.
- No watermark.
- No extra decorative elements unrelated to evaluation logic.

---

## Optional Enhanced Prompt (English, more detailed)

Design a professional white-background 16:9 technical infographic for a machine-learning evaluation framework called "Safety-Critical Eval for nuScenes". Use **#ff9300** as the dominant accent color for titles, flow arrows, and critical metric blocks. Use black for text and light gray for subtle structure lines.

Show a six-stage pipeline from left to right:
1. Prediction file input (`results_nusc.json`)
2. Ground-truth + metadata loading
3. Safety filtering (`class_range` and global `safety_max_dist`)
4. Detection matching (center-distance based)
5. Weighted metric engine (weighted mAP, weighted scale error by w/l/h, orientation and velocity errors weighted by ego-object yaw difference in two angle bins)
6. Output reports (`Safety-Critical NDS`, `metrics_summary.json`, `metrics_details.json`)

Add a compact "Why Better for Safety?" section with 4 short points:
- Focus on near-field risk
- Higher impact for vulnerable classes
- More interpretable scale error dimensions
- Context-aware orientation/velocity evaluation

Keep typography clean and modern, use rounded rectangles and thin connectors, preserve generous whitespace, and maintain a strictly minimal corporate style.

