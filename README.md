# EI-Driven Digital Tool for Faculty Motivation — Experiments & Prototype

A feasibility / design-science study and reference pipeline for an
emotion-intelligence-driven digital tool aimed at motivating **higher
education faculty** (ППС), built and validated on public datasets only
(no human-subject data collection).

## 1. What this is

The pipeline has three parts:

1. **Emotion perception module** — classifies the emotion expressed in a
   short text (a faculty check-in, journal entry, feedback comment, etc.).
2. **SDT need-mapping module** — maps the detected emotion to one of the
   three basic needs from Self-Determination Theory (Deci & Ryan):
   autonomy, competence, relatedness.
3. **Intervention generator** — produces a short, personalized motivational
   message targeted at that need (template-based by default; optionally via
   an LLM call).

This is packaged as a research pipeline with three experiments, not as a
finished chatbot product.

## 2. Why this is a genuine gap (not just "another EI chatbot")

Most existing work on emotion-aware AI / chatbots in higher education
targets **students**, not faculty. Where teacher emotions have been studied
computationally, it has been (a) about school (K-12) teachers, not
higher-ed faculty, and (b) descriptive rather than intervention-oriented:

- Chen, Shi, Zhang & Qu (2020, *Frontiers in Psychology*) analysed about a
  million K-12 teacher forum posts using **word-frequency lexicon counting**
  (8 discrete emotions) — a descriptive study, no intervention tool.
- Li (2022) analysed sentiment in **student evaluations of teaching** using
  a BiLSTM model — emotions *about* the teacher, not the teacher's own
  emotional state, and again no intervention tool.

This repository explicitly builds on and is benchmarked against that first,
simpler line of work (the lexicon-counting approach is reproduced here as
the `nrc_lexicon` baseline) while extending it in three ways: (1) target
population is higher-ed faculty, not school teachers; (2) it closes the
loop from "detect emotion" to a theory-grounded, personalized intervention;
(3) it explicitly tests cross-dataset / cross-domain generalization, which
the prior work does not.

## 3. What this does NOT claim

No public, labeled dataset of real higher-education faculty emotions exists
(checked before starting this project). So this study:

- **Does** show how accurately public emotion datasets generalize to
  higher-ed-relevant text (a domain-shift feasibility test).
- **Does** show that theory-driven personalized messages score
  significantly higher on an automatic relevance metric than a generic
  message.
- **Does NOT** show that real faculty would actually feel more motivated by
  this tool. That requires a follow-up study with real faculty (survey
  instruments such as WLEIS/TEIQue-SF for emotional intelligence and WTMST
  for teacher motivation, pre/post intervention, ethics approval). See
  "Future work" below.

## 4. Repository structure

```
.
├── data/
│   ├── raw/                 # downloaded datasets (gitignored, auto-fetched)
│   ├── lexicon/              # vendored NRC Emotion Lexicon (see its README)
│   └── vignettes/             # ORIGINAL: 75 illustrative faculty vignettes
│       ├── faculty_vignettes.csv
│       └── need_descriptions.csv
├── src/
│   ├── data_loader.py         # downloads + parses GoEmotions / ISEAR
│   ├── label_mapping.py       # GoEmotions(27) -> Ekman+neutral(7), ISEAR labels
│   ├── models.py               # NRCLexiconBaseline, TfidfBaseline, TransformerBaseline
│   ├── sdt_mapping.py           # emotion -> SDT need
│   ├── intervention_generator.py  # need -> motivational message
│   └── metrics.py               # classification metrics, TF-IDF relevance scorer
├── experiments/
│   ├── exp1_classification.py    # accuracy + cross-dataset generalization
│   ├── exp2_domain_shift.py       # near-domain test on faculty vignettes
│   └── exp3_intervention_quality.py  # generic vs personalized message quality
├── results/                      # metrics, predictions, figures (committed)
├── run_all.py                    # runs Exp.1 -> Exp.2 -> Exp.3
├── requirements.txt
└── INSTRUCTIONS_RU.md            # usage instructions in Russian
```

## 5. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The transformer baseline (`transformers`, `torch`) is optional; everything
else runs with just the core dependencies.

## 6. Running

```bash
python run_all.py --quick     # fast smoke test (~4000 training rows, ~10s)
python run_all.py             # full run on the full GoEmotions training set
python run_all.py --use_transformer   # also fine-tune a transformer in Exp.1
                                        # (needs internet access to the
                                        # Hugging Face Hub)
python run_all.py --use_llm           # also try an LLM few-shot classifier
                                        # in Exp.1 (subsampled) and Exp.2
                                        # (full 75 vignettes); needs
                                        # ANTHROPIC_API_KEY and costs real
                                        # API calls (cached on disk, see
                                        # results/.llm_cache.json)
```

Each experiment can also be run on its own, e.g. `python
experiments/exp1_classification.py --quick`.

**Cost/time estimate for `--use_llm` at the default settings** (150/class
stratified sample × 7 classes in Exp.1 in-domain, ×5 classes cross-dataset,
+ 75 vignettes in Exp.2; 5 few-shot examples/class ≈ 1,730 API calls,
≈1.7M input tokens total):

| model | est. cost | est. time (8 parallel workers) |
|---|:---:|:---:|
| `claude-sonnet-4-6` (default) | ≈ $5 | ≈ 6-7 min |
| `claude-haiku-4-5-20251001` (set via `ANTHROPIC_MODEL`) | ≈ $1.5-2 | ≈ 4-5 min |

These are estimates from token-count math, not a live invoice — check your
actual usage in the console. Lower `--llm_sample_size` or
`--llm_examples_per_label` to spend less; results are cached, so a
re-run after an interruption only pays for the calls not yet completed.

Datasets are downloaded automatically on first run from public GitHub
mirrors and cached in `data/raw/` (≈4 MB total). If your machine has no
internet access at all, download them elsewhere and place them manually at
the paths printed by `src/data_loader.py`.

## 7. Experiments

**Experiment 1 — classification accuracy & cross-dataset generalization.**
Trains each baseline on GoEmotions (Demszky et al., 2020), reduced to the
Ekman-6 + neutral grouping, and evaluates it both in-domain (GoEmotions
test split) and cross-dataset on ISEAR (Scherer & Wallbott), restricted to
the 5 labels shared with GoEmotions (anger, disgust, fear, joy, sadness).

**Experiment 2 — domain shift on faculty vignettes.** Applies the same
trained classifiers to 75 original, illustrative faculty vignettes
(`data/vignettes/faculty_vignettes.csv`), each one operationalizing a
stressor category reported in the qualitative literature on academic-staff
burnout/motivation (workload, recognition, autonomy/bureaucracy, isolation,
evaluation anxiety, career stagnation, work-life imbalance, positive
engagement), 8 vignettes per emotion class.

**Experiment 3 — intervention message quality.** For each vignette,
generates (a) one fixed generic message and (b) a personalized message
based on the SDT need mapped from the (gold-labeled) emotion, then scores
both against a canonical description of the target need using TF-IDF
cosine similarity, plus a paired t-test / Wilcoxon signed-rank test. Also
exports a blank rubric (`results/exp3_rubric_template.csv`) for optional
human expert scoring.

**Optional 4th classifier (`--use_llm`) — LLM few-shot.** Instead of
keyword/n-gram matching, classifies via an LLM API call with 5 labeled
in-context examples per class (`--llm_examples_per_label`). Skipped
automatically without `ANTHROPIC_API_KEY`. Costs one API call per
*predicted* example (responses are cached in `results/.llm_cache.json`,
calls run in parallel via `--llm_workers`, default 8), so Exp.1 evaluates
it on a **stratified** sample (`--llm_sample_size`, default 150 rows per
class — not a flat random sample, so rare classes like `disgust`/`fear` in
GoEmotions are not left with only 2-3 examples) rather than the full test
sets, while Exp.2 always runs it on the full 75 vignettes.

## 8. Results (full run, no subsampling)

Classifier accuracy across an increasing domain-shift ladder, with the
majority-class ("always predict the most frequent label") baseline for
reference — without it, "38% accuracy" is impossible to judge correctly:

| model | majority-class baseline | GoEmotions (in-domain) | ISEAR (cross-dataset) | Faculty vignettes (near-domain) |
|---|:---:|:---:|:---:|:---:|
| (always predict the most frequent class) | 0.350 / 0.200 / 0.200 | — | — | — |
| `nrc_lexicon`  | — | 0.384 | 0.326 | 0.400 |
| `tfidf_logreg` | — | 0.599 | 0.259 | 0.225 |

(majority-class baseline: 0.350 for GoEmotions' 7 imbalanced classes, 0.200
for ISEAR's and the vignettes' 5 balanced classes)

Read against that baseline: `tfidf_logreg` clearly beats chance in-domain
(0.599 vs 0.350) but is barely above chance once the domain shifts (0.259
and 0.225 vs 0.200). `nrc_lexicon` is the opposite pattern: weaker
in-domain, but the only one of the two that stays meaningfully above
chance once the domain shifts (0.326 and 0.400 vs 0.200). The model that
"wins" in-domain is not the model you would want to actually deploy on
faculty text — a finding about domain generalization, not a broken
pipeline.

Sadness (linked to the relatedness need — isolation, lack of recognition)
is the hardest emotion to detect on the faculty vignettes for both models
(12% accuracy, barely above what you'd expect from random within that
class). Inspecting the actual misclassifications shows two distinct,
explainable failure modes: `tfidf_logreg` defaults to "neutral" whenever
the text lacks the informal/Reddit-style markers it learned to associate
with anger/disgust (5/8 anger and 5/8 disgust vignettes predicted
"neutral"); `nrc_lexicon` over-predicts "anger" for fear/sadness vignettes
(5/8 and 6/8 respectively) because many negative-valence words in the NRC
lexicon are tagged with anger *together with* fear/sadness, so anger wins
the word-count vote. Neither failure mode is really about model capacity —
both are about the (mismatched) training distribution.

Intervention message relevance (`exp3`, TF-IDF similarity to the target
need description, n = 75 vignettes):

| message type | mean similarity |
|---|:---:|
| generic (baseline) | 0.023 |
| personalized (EI + SDT) | 0.057 |

paired t-test t = 6.81, p < 0.0001; Wilcoxon signed-rank p = 0.0001.

Full metrics (per-class precision/recall/F1, confusion matrices,
per-vignette predictions) are in `results/`.

## 9. On model choice: why not just use a fancier architecture?

It is tempting to read "40% accuracy" as "use a better model" and reach for
something like a Kolmogorov-Arnold Network (KAN/pyKAN). That is very
unlikely to help here, for a concrete reason: KANs replace fixed
activation functions with learnable splines, which is a real advantage for
fitting smooth, low-dimensional continuous functions (the symbolic
regression / scientific-computing problems they were introduced for). Text
classification is the opposite kind of problem — high-dimensional, sparse,
discrete token input — and there is no published evidence that KANs help
there; if anything they are harder to train and less mature for this data
type. The bottleneck demonstrated by Exp.1/Exp.2 is **domain mismatch**
(training distribution vs. the target text register), not insufficient
model expressiveness, so a more expressive function approximator does not
address it.

What would plausibly help, roughly in order of expected impact:

1. **Real or higher-fidelity in-domain data.** The only way to reliably get
   high accuracy *specifically on faculty text* is to have labeled
   examples from that text register. This is exactly the gap this
   feasibility study identified — and the natural next, separately funded
   empirical step (real faculty data collection, see Limitations below).
2. **LLM-based few-shot classification** (`--use_llm` in this repo).
   Because it draws on broad semantic/contextual knowledge rather than
   surface keyword overlap, it is the most likely of the available options
   to catch *implied* emotion (e.g. quiet isolation phrased without any
   "sad" word), which is exactly where the current baselines fail hardest.
   This is genuinely testable: run `--use_llm` with your own
   `ANTHROPIC_API_KEY` and compare against the table above.
3. **A fine-tuned transformer** (`--use_transformer`). Likely improves the
   in-domain GoEmotions number somewhat over `tfidf_logreg` (published
   GoEmotions benchmarks with transformers are meaningfully but not
   dramatically higher), and may generalize a bit better cross-domain than
   raw TF-IDF n-grams because it carries pretrained semantic
   representations rather than purely lexical ones — but it is not
   expected to close the domain gap on its own.

None of these change the central methodological point: a classifier's
in-domain accuracy is not a safe proxy for how it will perform on a
different population, and that gap should be measured and reported, not
assumed away by picking a bigger model.

## 10. Limitations & future work

- No real faculty data was used or collected; Experiments 2-3 use
  illustrative, researcher-authored vignettes, not field data.
- The emotion→need mapping (`src/sdt_mapping.py`) is a literature-informed
  proposal, not an empirically validated instrument — it should be reviewed
  by domain experts (e.g. organisational psychologists) before being used
  with real people.
- The automatic relevance metric (TF-IDF similarity) is a proxy for message
  quality, not a measure of actual motivational impact.
- Next step: an empirical pilot with real faculty, using validated scales
  (WLEIS or TEIQue-SF for emotional intelligence, WTMST for teacher
  motivation), with ethics committee approval and informed consent.

## 11. Datasets & attribution

- **GoEmotions**: Demszky, D., Nemoto, K., Briakou, E., Yenidogan, M.,
  Sharma, S., Cowen, A., Nemenman, I., & Ravi, S. (2020). GoEmotions: A
  Dataset of Fine-Grained Emotions. *ACL 2020*.
- **ISEAR**: Scherer, K. R., & Wallbott, H. G. International Survey on
  Emotion Antecedents and Reactions.
- **NRC Emotion Lexicon (EmoLex)**: Mohammad, S. M., & Turney, P. D.
  (2013). Crowdsourcing a word-emotion association lexicon. *Computational
  Intelligence*, 29(3), 436-465. See `data/lexicon/README.md` for license
  notes — free for research use.

## 12. License

Code in this repository is released under the MIT License (see `LICENSE`).
The third-party datasets/lexicon are not covered by that license — see
Section 11 above.

---
Usage instructions in Russian: see `INSTRUCTIONS_RU.md`.
