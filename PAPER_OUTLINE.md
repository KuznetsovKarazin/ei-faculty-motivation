# Proposed Manuscript Structure

Target venue: MDPI *Trends in Higher Education* (Scopus/ESCI, CiteScore Q1 Social Sciences (misc)).
Likely article type: **Article** (full original research) or **Concept Paper** if the editor prefers a
shorter, more design-oriented framing — recommend submitting as a full Article given the amount of
empirical work (3 experiments, 6 classifiers, 75 scenarios, 21 human raters).

**Recommended framing decision (read this first):** this repository accumulated genuine NLP-benchmarking
depth (6 classifiers, fair-comparison fixes, a 4-level ablation) alongside an Education-feasibility
design (SDT theory, vignettes, human evaluation). **Pick the Education/feasibility framing as primary**
— it matches the target venue, and the strongest, most defensible finding (Section 6.4/6.5 of the
README) is a theory-grounded design finding, not a classifier-benchmarking one. Keep the classifier
comparison as supporting methodology (an honest answer to "could you just use an off-the-shelf model"),
not as the headline contribution. Do not write "we built an emotionally intelligent chatbot" — write
"we experimentally decompose personalization into its components and show theory-guided need modeling
contributes more to intervention quality than emotion recognition alone."

---

## Title (pick one, or merge)

1. *Theory-Guided Personalization Outperforms Emotion Recognition Alone: A Feasibility Study of an
   SDT-Grounded Digital Motivational Tool for Higher Education Faculty*
2. *From Detecting Emotion to Understanding Need: An Ablation Study of a Self-Determination-Theory-Based
   Digital Support Tool for Academic Staff*
3. *Can Public Datasets Support a Digital Motivational Tool for University Faculty? A Design-Science
   Feasibility Study with Human Expert Validation*

## Keywords

emotional intelligence; digital intervention; faculty motivation; Self-Determination Theory; higher
education; natural language processing; large language models; design science research; human-AI
evaluation

## Abstract (~200-250 words for MDPI)

Use the README Abstract as a draft base; trim to MDPI's typical structured-but-unlabeled abstract style:
1 sentence motivation/gap, 1-2 sentences method, 2-3 sentences key results (classifier ladder + ablation
+ human eval contrast), 1 sentence implication. Avoid the phrase "chatbot" in the abstract; use "digital
intervention tool" or "personalization pipeline."

## 1. Introduction

- Open on the *practical* problem: faculty motivation/burnout as an HR and institutional-performance
  issue (cite general burnout/WTMST literature), not on AI.
- State the gap precisely: EI-driven digital tools target students; faculty EI is measured, not acted
  on (cite Chen et al. 2020, Li 2022 as the two closest prior computational works, both reviewed in
  README Section 1).
- State the central research question as a contribution claim, not a build claim:
  *"In faculty-support systems, does theory-guided need modeling contribute more to intervention quality
  than emotion recognition alone, and does this hold once classification is realistically imperfect?"*
- Briefly preview the three studies (classifier feasibility, ablation, human validation) and the
  headline finding.
- End with explicit contributions, numbered (3-4 bullets): (1) gap identification, (2) a label-space-fair
  benchmarking methodology across 4 classifier families applicable to anyone building this kind of tool
  on public data, (3) a 4-level personalization ablation design, (4) blind human validation showing
  automatic text metrics misjudge generated intervention quality.

## 2. Theoretical Background

- 2.1 Emotional intelligence in educational AI — brief, citing the student-focused literature this study
  diverges from.
- 2.2 Self-Determination Theory and teacher motivation — Deci & Ryan (2000); WTMST (Fernet et al., 2008);
  autonomy/competence/relatedness as the mapping target for detected emotion.
- 2.3 Computational emotion recognition — Ekman/GoEmotions taxonomy, lexicon vs. supervised vs. LLM
  approaches, domain shift as a known NLP problem (cite generally, e.g. domain adaptation literature) —
  set up why a fair cross-domain comparison matters.
- 2.4 Gap statement, explicit: no prior work combines (a) higher-ed faculty as the target population,
  (b) a dynamic, classifier-driven emotional signal (not a static EI questionnaire), (c) theory-routed
  personalization, and (d) human validation of what that personalization actually adds.

## 3. Materials and Methods

- 3.1 Design — Design Science Research framing (cite Hevner or Peffers DSR framework briefly); overall
  pipeline diagram (consider adding a simple architecture figure: text -> emotion classifier -> SDT need
  -> message generator -> [oracle/end-to-end] -> evaluation).
- 3.2 Datasets — GoEmotions, ISEAR (with citations); explicitly state why no faculty-emotion dataset
  exists and that this is a known, named gap, not an oversight.
- 3.3 Emotion classifiers — all 4 families (README Section 5/6.1), explicitly describing the label-space
  fairness fix as a methodological contribution, not a footnote.
- 3.4 Faculty scenario set — construction method (8 stressor categories from the qualitative
  burnout/WTMST literature), 75 items, 15/class; explicitly labeled a stress-test, not a dataset of real
  faculty; mention `vignette_validation_template.csv` as available-but-not-yet-completed face validity
  check (report honestly if still not done at submission time, or report it if you complete it before
  submission — recommended).
- 3.5 SDT need mapping & intervention generation — the emotion->need table with rationale
  (`src/sdt_mapping.py`), and the 4 ablation levels with example messages (give 1-2 real, anonymized
  example messages per level in a table, pulled from `results/exp3_outputs.csv`).
- 3.6 Automatic evaluation — TF-IDF similarity proxy, explicitly flagged as a secondary/sanity-check
  metric, not the primary outcome (justify briefly with the 6.3 finding as forward reference, or move
  that explanation here).
- 3.7 Human evaluation protocol — 21 raters' roles/experience (Table), blind A/B/C/D randomization
  procedure, 4 rating dimensions with operational definitions, stratified 25-item subsample rationale,
  Google Forms collection procedure. **State plainly that this was an expert content-validity rating
  exercise on synthetic text, not a human-subjects study on real people's emotional states** — note
  whether your institution requires/waived ethics review for this kind of expert-rating task (check with
  your IRB/ethics office and state the determination explicitly; do not assume exemption without
  checking).
- 3.8 Statistical analysis — paired t-tests + Wilcoxon signed-rank for the staircase comparisons, Cohen's
  d for effect size, pairwise Pearson r for inter-rater agreement (flag in Limitations that ICC would be
  preferable), majority-class baseline for classification accuracy context.

## 4. Results

Mirror README Section 6, in this order:
- 4.1 Classifier domain-shift ladder (Table + Figure `domain_shift_ladder.png`) + the label-space-fairness
  finding + the "better in-domain fit, worse out-of-domain calibration" finding (transformer's 54.4%
  out-of-label-space rate) as a standalone reported result, not just a footnote.
- 4.2 Per-emotion error analysis (sadness/disgust patterns from README 6.1 and earlier per-vignette
  inspection) — qualitative + quantitative.
- 4.3 Automatic ablation results (Table from README 6.2).
- 4.4 Human evaluation results — **lead figure of the whole paper**: `human_rubric_staircase.png` +
  Table from README 6.4. Report the emotion_only -> need_only dimension-by-dimension split (significant
  on 3/4, not relevance) as a nuanced, not overclaimed, finding.
- 4.5 Automatic-vs-human divergence as its own explicitly labeled result (README 6.3), not buried in
  Discussion — reviewers reward stating an unexpected/nuanced result plainly in Results.

## 5. Discussion

- Open with the headline finding restated as a generalizable claim (not just "our tool worked"):
  *theory-guided personalization, not emotion detection, is the active ingredient* — tie back to SDT
  literature on why need-satisfaction language should outperform mere affect-labeling.
  - Tie the largest jump (need_only -> full_context) to literature on specificity/concreteness in
    supportive communication if available (social support / health communication literature is a
    reasonable adjacent field to cite here).
- Methodological discussion: why fair label-space comparison changed the classifier-method conclusion;
  why automatic text metrics diverged from human judgment, and what that implies for anyone evaluating
  generative systems in this space (cite general LLM-evaluation-metric critique literature if available).
- Practical implications for higher-ed HR/management: a low-cost, theory-grounded design pattern
  institutions could adapt, with an explicit caveat about deployment readiness (low classifier accuracy
  in absolute terms, Section 4.1/4.2) - position as informing a deployable system's design, not as a
  deployable system itself.
- Explicitly connect to the *Trends in Higher Education* "Leadership and Management" framing: digital
  transformation, organizational resilience, HR innovation for academic staff support.

## 6. Limitations

Use README Section 7 nearly verbatim, plus:
- Synthetic scenario set (not real faculty); inter-rater agreement at the item level was low (report
  honestly, recommend ICC for any follow-up); single-institution/non-representative rater sample
  (describe rater recruitment honestly); automatic metric limitations (6.3) generalize as a caution about
  evaluating generative text quality broadly, not just within this study.

## 7. Conclusion

3-5 sentences: restate the question, the answer (theory-routing > emotion detection, confirmed by blind
human raters but invisible to automatic metrics), and the concrete next step (empirical pilot with real
faculty using WLEIS/TEIQue-SF + WTMST, ethics-approved).

## Standard MDPI back-matter sections

- **Author Contributions** (CRediT taxonomy — assign explicitly per author, including Kazakh
  collaborators; conceptualization / methodology / software / validation / formal analysis /
  investigation / writing).
- **Funding** — state the Kazakh grant explicitly here.
- **Institutional Review Board Statement** — state your actual determination (exempt / not applicable /
  approved with protocol number) once checked with your ethics office; do not leave implicit.
- **Informed Consent Statement** — for the 21 expert raters, confirm they consented to anonymized
  research use of their ratings (a one-line consent notice on the Google Form is standard practice -
  confirm this was included, or add a sentence to the form/methods describing what raters were told).
- **Data Availability Statement** — point to the GitHub repository (this one) and/or a Zenodo DOI if you
  archive it there (recommended for a citable, permanent record — Zenodo integrates with GitHub releases).
- **Acknowledgments** — datasets/lexicon attribution (README Section 8), the 21 raters (collectively,
  anonymized), Anthropic API usage disclosure if required by the journal's AI-use policy (check current
  MDPI policy on AI tool disclosure for both writing assistance and any AI-generated content such as the
  `full_context` messages themselves — this almost certainly needs an explicit methods-level disclosure
  given LLM-generated text is literally part of the experimental material, not just a writing aid).
- **Conflicts of Interest** — standard statement.

## Figures/Tables checklist for submission

- [ ] Figure 1: pipeline/architecture diagram (not yet created — recommend adding one, simple box-and-arrow)
- [ ] Figure 2: `domain_shift_ladder.png`
- [ ] Figure 3: confusion matrix panel (select 2-3 most informative from `results/figures/`, not all 19)
- [ ] Figure 4: `exp3_ablation_oracle.png` and/or `exp3_ablation_end_to_end.png`
- [ ] Figure 5: `human_rubric_staircase.png` (the lead figure)
- [ ] Table 1: classifier domain-shift ladder (README 6.1)
- [ ] Table 2: rater demographics (role x experience, from `human_eval_raw_responses.xlsx`)
- [ ] Table 3: human evaluation staircase (README 6.4)
- [ ] Supplementary Table: full per-vignette predictions (`exp2_predictions.csv`) and per-vignette
      message outputs (`exp3_outputs.csv`) as supplementary material / Zenodo deposit, not in-text.
