# M4 track findings (Nicholas)

Notebook: `m4_nicholas.ipynb`. Run on an RTX 4070 laptop GPU, 7 Nov 2026. Validation only, unless marked *test*.
All models use the shared pipeline (same data, tokenizer, vocabulary, evaluation, CRF, char-CNN and training loop as `full_version_resa.ipynb`). The reference model has exactly the same 4,585,110 parameters as Resa's Stage 2 model.

## 1. Validation results (3 seeds each: 42, 43, 44)

Score = mean of intent accuracy and slot micro F1. Keep margin = 0.51 (2 × sqrt(2/3) × pooled seed std 0.31).

| Model | Params | Intent | Slot F1 | Score (mean ± std) | vs | Gain | Beats margin |
|---|---|---|---|---|---|---|---|
| R: BiLSTM-CRF + char-CNN (reference) | 4.59M | 87.41 | 81.29 | 84.35 ± 0.53 | | | |
| M4-1L: Co-Interactive, label attention, 1 layer | 6.03M | 87.34 | 80.99 | 84.16 ± 0.40 | R | −0.18 | no |
| M4-2L: 2 co-interactive layers | 7.35M | 86.70 | 79.13 | 82.92 ± 0.29 | R | −1.43 | no |
| M4-noLA: 1 layer, no label attention | 6.12M | 86.98 | 79.81 | 83.40 ± 0.15 | R | −0.95 | no |
| SP: Stack-Propagation | 4.60M | 87.67 | 81.79 | 84.73 ± 0.10 | R | +0.38 | no |
| SP + reg (label smoothing 0.1, count word dropout α=0.75) | 4.60M | 87.10 | 81.51 | 84.30 ± 0.19 | R + reg | −0.55 | no |
| R + reg | 4.59M | 87.44 | 82.26 | 84.85 ± 0.32 | R | +0.50 | no (just under 0.51) |

Selected by the rule: **R**. Official run (seed 42, epoch 36): validation 86.82 / 81.75, *test 85.94 intent / 79.19 slot F1*.

## 2. What we learned

1. **The Co-Interactive model (M4) does not beat the reference.** With one layer it is within noise (−0.18). Adding a second layer (−1.43) or removing the label attention (−0.95) makes it clearly worse. More interaction plus more parameters on 11.5k training utterances hurts.
2. **M4 reshuffles intent errors instead of reducing them.** Compared with R (both seed 42), M4 fixes 64 intent errors but introduces 61 new ones: net +3 of 2,033.
3. **The mechanism works as designed, which is good for the report.** In the attention figure (`runs/m4/figures/m4_intent_attention.png`), the intent stream reads the slot words. Examples:
   - "**loop** this track": `player_setting` points to `music_settings` (R said `play_music`).
   - "set the volume **to low**": `change_amount` points to `audio_volume_down` (R said `audio_volume_up`).
4. **Bidirectional coupling also spreads errors in both directions.** When the intent is wrong, the slot types follow it, and vice versa:
   - "please play leviathan": M4 predicts `play_audiobook` *and* tags `audiobook_name` (gold: `play_music`, `song_name`).
   - "never ever play me any country": `play_game` *and* `game_name`.

   Slot precision drops from 82.49 to 79.20. M4 predicts more spurious spans (186 vs 134) and more wrong-type spans (77 vs 56). F1 falls most for `player_setting` (−18.8) and `song_name` (−13.2).
5. **Stack-Propagation is the best interaction variant:** highest mean intent accuracy (87.67) and the most stable (std 0.10). Its +0.38 gain is still under the margin.
6. **The two regularisers help the reference (+0.50, highest mean score overall) but not Stack-Propagation.** This matches Evan's finding. The gain sits right at the margin, so it is not conclusive with 3 seeds.
7. **Many remaining intent errors look hard to fix without pretraining.** Both models get 204 dev utterances wrong, 142 of them with the same wrong label. About 72% of errors cross scenarios (`qa_factoid` ↔ `general_quirky` is the top pair). Some look like label noise ("phones" is gold `alarm_remove`; "by proper formula" is gold `weather_query`).

## 3. Caveats

- M4 used the reference's training recipe (lr 1e-3, no warm-up). Attention layers trained from scratch often want a warm-up, so M4 was not tuned for itself. Two of the six 2-layer / no-label-attention runs were still at their best score when they hit the 60-epoch cap.
- Each result uses 3 seeds. Differences below about 0.5 points are not reliable.
- Section 9 numbers (fixed/broken counts, slot error kinds) come from reloading the seed-42 checkpoints. M4's recomputed validation score is 83.76 vs 83.75 logged; the 0.01 difference is CPU vs GPU arithmetic.

## 4. Suggestions for the final Stage 2 system

- Keep the BiLSTM-CRF + char-CNN + LR decay as the Stage 2 model.
- Decide on validation whether to add label smoothing + count-based word dropout (R + reg). Two more seeds (45, 46) for R, R + reg and SP would settle it in about 45 minutes on a GPU.
- Report M4 as an unsuccessful experiment, with the coupling explanation above. It is direct evidence for the "analysis and lessons learned" section.
- Error cases for the report: "please play leviathan" (coupled intent + slot error), "set the volume to low" (fixed by slot evidence), "phones" → `alarm_remove` (no signal or label noise).
