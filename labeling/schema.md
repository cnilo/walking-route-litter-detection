# Labeling schema for image-level litter annotation

This document describes the guidelines used by the research team to annotate
the presence of litter in point-of-view (POV) images captured during the
walking routes in downtown Santiago, Chile.

The goal of the annotation is to obtain a **binary, image-level label** that
indicates whether **any visible litter** is present in the scene.

- `truth = 1` → at least one instance of litter is visible in the image  
- `truth = 0` → no litter is visible in the image  

These labels are used as **ground truth** for evaluating the YOLOv8-based
detector and for computing the TRUTH, TP, TN, FP and FN variables in the
analysis.

---

## 1. Definition of litter

For the purpose of this project, **litter** is defined as:

> Any discarded, misplaced or abandoned item of waste or packaging
> (e.g. paper, plastic, cardboard, cans, bottles, food containers, bags,
> cigarette butts, etc.) that is **not in its intended functional place**
> (e.g. on the ground, on street furniture, stuck to surfaces) and is
> **not being actively used**.

Key aspects of this definition:

- **Discarded / abandoned**: the object appears to have been left behind
  and is no longer in use.
- **Misplaced**: the object is located where it is clearly not supposed
  to be (e.g. on the sidewalk, street, grass, stairs, benches).
- **Not functional**: the object is not clearly serving an ongoing,
  legitimate function in the context of the scene (e.g. not part of a
  display, equipment, or infrastructure).

---

## 2. Positive cases (`truth = 1`)

Annotators must assign `truth = 1` when **at least one** of the following
cases is visible in the image:

1. **Loose waste on the ground**
   - Food wrappers, cups, straws, napkins
   - Plastic or paper bags
   - Beverage cans or bottles
   - Cardboard pieces, boxes, trays
   - Cigarette butts, matches

2. **Waste on street furniture or urban elements**
   - Items left on benches, steps, walls, fences, bollards, etc.
   - Items stuck or wedged into grids, railings, or similar elements.

3. **Overflowing bins where waste has fallen outside**
   - Litter on the ground around the bin counts as litter.
   - The simple fact that the bin is full does **not** count; only
     the items that are outside the container.

4. **Accumulations of clearly discarded material**
   - Small piles of mixed trash, even if not individually separable,
     as long as they are recognizable as waste.

In all cases, **even a single visible item** is sufficient to label the
image with `truth = 1`.

---

## 3. Negative cases (`truth = 0`)

Annotators must assign `truth = 0` when **no litter** as defined above is
visible, including the following cases:

1. **Natural elements**
   - Dry leaves, branches, soil, mud, stones.
   - Flower petals, fruit, seeds falling from trees.

2. **Infrastructure and fixed objects**
   - Street furniture (benches, bins, bollards, planters, signs).
   - Construction materials that are clearly in use and organized
     (e.g. stacked bricks at an active work site).

3. **Objects that appear to be in functional use**
   - Merchandise displayed by street vendors.
   - Tools, boxes or carts being actively used by workers.
   - Deliveries that are obviously in process (e.g. stacked packages
     next to an open truck with workers nearby).

4. **Waste inside containers**
   - Trash that is fully contained inside a bin or bag and not
     spilling out does **not** count as litter for the purpose of
     the image-level label.

5. **Unclear micro-elements**
   - Very small or ambiguous spots that cannot be confidently
     identified as litter at the available resolution.

If annotators are uncertain and cannot reliably confirm the presence of
litter, they should default to `truth = 0`.

---

## 4. Ambiguous cases and use of context

Ambiguous cases must be resolved by **considering the object in context**:

- Ask whether the object appears to be:
  - discarded/abandoned, **and**
  - in a location where it clearly does **not** serve a current function.

Examples:

- A cardboard box next to a store entrance:
  - If it is open, with items being moved in/out → **likely functional** → `truth = 0`.
  - If it is crushed and alone on the sidewalk, away from any activity → **likely litter** → `truth = 1`.

- Plastic bags near a street vendor:
  - If they clearly contain merchandise or are part of the stall setup → **functional** → `truth = 0`.
  - If they are empty or crumpled on the ground, apart from the stall → **litter** → `truth = 1`.

- Items near waste collection points:
  - Clearly stacked containers waiting for collection → `truth = 0`.
  - Dispersed waste on the pavement after collection → `truth = 1`.

Annotators are encouraged to use all available contextual clues in the
image (people, activities, vehicles, signage) to decide whether an object
is likely to be litter or part of normal, functional use of space.

---

## 5. Annotation procedure

1. **Image-level labeling**
   - Each image is labeled with a single binary value:
     - `truth = 1` if one or more items of litter are visible.
     - `truth = 0` otherwise.

2. **Double coding (when applicable)**
   - A subset of images is independently labeled by at least two
     members of the research team.
   - Discrepancies are discussed and resolved by consensus, using
     the definitions in this document.

This schema is intentionally designed to be **transparent and reproducible**,
while respecting privacy and ethical constraints by not releasing the full
set of annotated images.
