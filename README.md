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
│   └── vignettes/             # ORIGINAL: 40 illustrative faculty vignettes
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
```

Each experiment can also be run on its own, e.g. `python
experiments/exp1_classification.py --quick`.

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
trained classifiers to 40 original, illustrative faculty vignettes
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

## 8. Results (full run, no subsampling)

Classifier accuracy across an increasing domain-shift ladder:

| model         | GoEmotions (in-domain) | ISEAR (cross-dataset) | Faculty vignettes (near-domain) |
|---------------|:---:|:---:|:---:|
| `nrc_lexicon`  | 0.384 | 0.326 | 0.400 |
| `tfidf_logreg` | 0.599 | 0.259 | 0.225 |

The more sophisticated, purely in-domain-trained classifier
(`tfidf_logreg`) is clearly better **in-domain**, but generalizes **worse**
than the much simpler lexicon-counting baseline both cross-dataset and on
the faculty vignettes. This is itself a relevant finding for anyone
building this kind of tool cheaply on public data: in-domain accuracy is
not a safe proxy for real-world (out-of-domain) performance, and a
hybrid/ensemble or few-shot LLM-based approach is likely needed for
production use — see `--use_transformer` and the `llm` mode in
`intervention_generator.py` as starting points.

Sadness (linked to the relatedness need — isolation, lack of recognition)
is the hardest emotion to detect on the faculty vignettes for both models
(12% accuracy), which is a meaningful limitation given how central
isolation/recognition issues are in the faculty-burnout literature.

Intervention message relevance (`exp3`, TF-IDF similarity to the target
need description, n = 40 vignettes):

| message type | mean similarity |
|---|:---:|
| generic (baseline) | 0.023 |
| personalized (EI + SDT) | 0.057 |

paired t-test t = 6.81, p < 0.0001; Wilcoxon signed-rank p = 0.0001.

Full metrics (per-class precision/recall/F1, confusion matrices,
per-vignette predictions) are in `results/`.

## 9. Limitations & future work

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

## 10. Datasets & attribution

- **GoEmotions**: Demszky, D., Nemoto, K., Briakou, E., Yenidogan, M.,
  Sharma, S., Cowen, A., Nemenman, I., & Ravi, S. (2020). GoEmotions: A
  Dataset of Fine-Grained Emotions. *ACL 2020*.
- **ISEAR**: Scherer, K. R., & Wallbott, H. G. International Survey on
  Emotion Antecedents and Reactions.
- **NRC Emotion Lexicon (EmoLex)**: Mohammad, S. M., & Turney, P. D.
  (2013). Crowdsourcing a word-emotion association lexicon. *Computational
  Intelligence*, 29(3), 436-465. See `data/lexicon/README.md` for license
  notes — free for research use.

## 11. License

Code in this repository is released under the MIT License (see `LICENSE`).
The third-party datasets/lexicon are not covered by that license — see
Section 10 above.

---
Usage instructions in Russian: see `INSTRUCTIONS_RU.md`.
