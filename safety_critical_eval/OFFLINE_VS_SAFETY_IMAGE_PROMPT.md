# Offline vs Safety-Critical Comparison Diagram Prompt

## Primary Prompt (English)

Create a clean 16:9 side-by-side comparison infographic for "Offline Official nuScenes Eval vs Safety-Critical Eval".

Visual style requirements:
- White solid background only (pure white, no gradients, no textures).
- Primary theme color must be **#ff9300** for titles, section headers, key arrows, and emphasis blocks.
- Secondary neutral colors allowed only for support elements: black text, light gray borders (#d9d9d9), very light gray panels (#f7f7f7).
- Flat modern vector style, minimal and professional.
- Strong readability and balanced spacing.

Content/layout requirements:
- Two-column layout:
  - Left column: **Offline Official Eval**
  - Right column: **Safety-Critical Eval**
- For each column, include a compact pipeline with 4 boxes:
  1) Input
  2) Filtering
  3) Metric Computation
  4) Outputs
- Explicitly show key differences:
  - Filtering:
    - Official: `class_range`
    - Safety: `class_range + safety_max_dist`
  - mAP:
    - Official: class-average mAP
    - Safety: class-weighted mAP (`class_ap_weights`)
  - Errors:
    - Official: standard `scale/orient/vel`
    - Safety: weighted `scale_err` (w/l/h), yaw-aware weighted `orient_err`, yaw-aware weighted `vel_err`
  - Comparability:
    - Official: leaderboard-compatible
    - Safety: risk-prioritized internal metric
- Add a bottom "When to use which?" strip:
  - Official eval for benchmark reporting
  - Safety eval for risk-oriented optimization
- Use simple icons (file, filter, balance scale, shield, chart) with consistent style.
- All text in English, concise and technical.
- Ensure final image ratio is exactly 16:9.

Output constraints:
- No dark background.
- No photorealistic rendering.
- No watermark.
- No excessive decoration.

---

## Optional Enhanced Prompt (English, more detailed)

Design a white-background 16:9 technical comparison infographic titled "Official nuScenes Eval vs Safety-Critical Eval". Use **#ff9300** as the main accent color across headers, dividers, and highlighted differences.

Build a two-panel comparison:

Left panel: "Offline Official Eval"
- Standard nuScenes detection pipeline
- Official filtering and matching rules
- Standard mAP and TP errors
- Official NDS for public comparability

Right panel: "Safety-Critical Eval"
- Adds near-range prioritization (`safety_max_dist`)
- Uses class-priority weighting for mAP
- Uses weighted scale error across w/l/h
- Uses yaw-context weighted orientation and velocity errors
- Produces safety-oriented NDS-like score

Add a middle "Difference Matrix" with short rows:
- Distance handling
- Class aggregation
- Scale error definition
- Orientation/velocity context awareness
- External comparability

At the bottom, include a concise recommendation:
"Use both together: official metrics for standard reporting, safety-critical metrics for risk-driven engineering."

Keep typography clean, vector icons minimal, spacing generous, and maintain strict white background with 16:9 composition.

