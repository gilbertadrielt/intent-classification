# Stage 2 alternatives: findings (Nicholas)

Notebook: `stage2_alternatives_nicholas.ipynb`, run on an RTX 4070 laptop GPU on 8-9 Nov 2026. Each idea is **one change**
on Resa's Stage 2 model (R = BiLSTM + char-CNN + CRF + LR decay, 4,585,110 params). The pipeline, seeds and keep rule
are the same as in `m4_nicholas.ipynb`, and runs already done there were reused. Score = mean of dev intent accuracy
and slot micro F1.

## 1. One change at a time (3 seeds: 42, 43, 44; keep margin 0.50)

| Variant | What changes | Params | Intent | Slot F1 | Score | Gain vs R |
|---|---|---|---|---|---|---|
| R (Resa's Stage 2) | none | 4.59M | 87.41 | 81.29 | 84.35 | |
| SP: Stack-Propagation | every word votes on the intent; the vote feeds the slot tagger | 4.60M | 87.67 | 81.79 | 84.73 | +0.38 |
| A: BIOES tags | tags also mark span ends (E-) and one-word spans (S-) | 4.68M | 87.65 | 81.39 | 84.52 | +0.18 |
| B: Hierarchical intent | p(scenario) × p(intent \| scenario) | 4.59M | 87.00 | 82.00 | 84.50 | +0.15 |
| C: Compatibility decoding | slot types never seen with the predicted intent are forbidden | 4.59M | 87.46 | 81.52 | 84.49 | +0.14 |

No single change beats the 0.50 margin. Each one lands slightly above R, inside the noise.

## 2. Five-seed confirmation of the finalists (seeds 42-46; keep margin 0.35)

| Model | Intent | Slot F1 | Score (± std) | Gain vs R | Beats margin |
|---|---|---|---|---|---|
| R | 87.37 | 81.32 | 84.34 ± 0.40 | | |
| **R + reg** (label smoothing 0.1 + count-based word dropout α = 0.75) | 87.38 | 82.15 | **84.76 ± 0.26** | **+0.42** | **yes** |
| SP | 87.55 | 81.83 | 84.69 ± 0.09 | +0.34 | no (just under) |
| SP + reg | 87.16 | 81.79 | 84.47 ± 0.28 | +0.13 | no |

**Selected: R + reg**, i.e. Resa's model with Evan's two regularisers. It has zero extra parameters.
Official run (seed 42, epoch 24): dev 87.51 / 81.62, **test 86.31 intent / 79.79 slot F1**, 4,585,110 params.

## 3. What we learned

1. **The regularisers are the only change that clearly helps,** and the gain comes from slots (+0.83 slot F1, intent unchanged). Count-based word dropout hides rare words during training, so the model learns to tag spans from context.
2. **Stack-Propagation is the most stable model** (seed std 0.09, highest intent at 87.55) but just misses the margin. Adding the regularisers to it *hurts*, so the two do not stack.
3. **Ideas aimed at a specific error type barely move it:**
   - BIOES was meant to fix boundaries but leaves slot F1 unchanged (81.39 vs 81.29).
   - Hierarchical intent was meant to fix cross-scenario intent errors, but intent goes *down* (87.00) while slots go up.
   - Compatibility decoding gives the smallest gain, because a wrong intent now also forces wrong slot types (the same coupling problem as M4).
4. **The remaining errors are hard ones.** R + reg still makes 254 intent errors on dev, 182 of them (72%) across scenarios (top pairs: `general_quirky` ↔ `news_query` / `qa_factoid`). Where the intent is right but the slots are wrong, the errors are mostly genuinely ambiguous spans, e.g. "put on cannibal queen" (song vs artist) and "play chopin" (artist vs playlist).
5. **Single runs are noisy, even with the same seed.** SP with seed 42 scored 84.84 in `m4_nicholas` and 83.92 here (GPU non-determinism). With seed 42, R + reg vs R fixes 45 intents and breaks 45, and fixes 80 slot sets and breaks 78: the gain only shows up on average over seeds. Any comparison between single runs (including test numbers) should be read with about ±0.5 to 1 point of noise.

## 4. Suggestion for the team

Use **Resa's model + label smoothing + count-based word dropout** as the final Stage 2 system. This tells a clean story for the report:
- the diagnosis motivated char-CNN and the CRF (Resa)
- the regularisers come from Evan's experiments
- the 5-seed check confirms they help beyond noise
- M4, Stack-Propagation, BIOES, hierarchical intent and compatibility decoding are tried-and-failed ideas, each with a reason

Note: this run trained at about 55 s per epoch, against about 12 s in the M4 run, so the laptop was probably on battery or in power-saving mode. Plug in and use the "Best performance" power mode for the final `main.ipynb` run.
