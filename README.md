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
│   └── vignettes/             # ORIGINAL: 75-scenario researcher-authored set
│       ├── faculty_vignettes.csv
│       ├── need_descriptions.csv
│       └── vignette_validation_template.csv  # blank, for expert face-validity check
├── src/
│   ├── data_loader.py         # downloads + parses GoEmotions / ISEAR, stratified_sample
│   ├── label_mapping.py       # GoEmotions(27) -> Ekman+neutral(7), ISEAR labels
│   ├── models.py               # NRCLexiconBaseline, TfidfBaseline, TransformerBaseline, LLMFewShotBaseline
│   ├── sdt_mapping.py           # emotion -> SDT need
│   ├── intervention_generator.py  # need -> motivational message
│   ├── metrics.py               # classification metrics, out-of-label-space rate, TF-IDF relevance scorer
│   └── plotting.py               # confusion matrix, domain-shift ladder, paired comparison charts
├── experiments/
│   ├── exp1_classification.py    # accuracy + cross-dataset generalization + fair tfidf_5class
│   ├── exp2_domain_shift.py       # near-domain test on the vignette scenario set
│   ├── exp3_intervention_quality.py  # generic vs personalized, oracle + end-to-end, blind rubric export
│   └── exp3b_rubric_analysis.py   # analyzes filled-in blind rubrics from human raters
├── results/                      # metrics, predictions, figures (committed)
├── run_all.py                    # runs Exp.1 -> Exp.2 -> Exp.3 + domain-shift summary chart
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
mirrors and cached in `data/raw/` (≈4 MB total, not committed to the repo -
see "Datasets & attribution" for why). **If your machine/sandbox has no
internet access at all** (e.g. an isolated review environment), download
these 5 files manually on a machine that does have access, and place them
in `data/raw/` under the exact names below before running anything:

| save as | source URL |
|---|---|
| `data/raw/goemotions_train.tsv` | https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/train.tsv |
| `data/raw/goemotions_dev.tsv` | https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/dev.tsv |
| `data/raw/goemotions_test.tsv` | https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/test.tsv |
| `data/raw/goemotions_ekman_mapping.json` | https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/ekman_mapping.json |
| `data/raw/isear_raw.csv` | https://raw.githubusercontent.com/sinmaniphel/py_isear_dataset/master/isear.csv |

Once those 5 files exist locally, every script runs fully offline (no
network calls at all unless you pass `--use_llm`/`--use_transformer`).

## 7. Experiments

**Experiment 1 — classification accuracy & cross-dataset generalization.**
Trains each baseline on GoEmotions (Demszky et al., 2020), reduced to the
Ekman-6 + neutral grouping, and evaluates it both in-domain (GoEmotions
test split) and cross-dataset on ISEAR (Scherer & Wallbott), restricted to
the 5 labels shared with GoEmotions (anger, disgust, fear, joy, sadness).
Also trains a **label-space-matched** `tfidf_logreg_5class` variant (same
algorithm, trained only on the 5 ISEAR-shared classes) for a fair
comparison with `nrc_lexicon`/`llm_fewshot` — see "Fair comparison" below,
this matters a lot for the conclusions.

**Experiment 2 — domain shift on a researcher-authored scenario set.**
Applies the same trained classifiers (incl. the fair `tfidf_logreg_5class`
variant) to 75 original, illustrative faculty vignettes
(`data/vignettes/faculty_vignettes.csv`), each one operationalizing a
stressor category reported in the qualitative literature on academic-staff
burnout/motivation (workload, recognition, autonomy/bureaucracy, isolation,
evaluation anxiety, career stagnation, work-life imbalance, positive
engagement), 15 vignettes per emotion class. **This is a constructed
scenario set / vignette-based stress test, not a dataset of real faculty
emotions** - do not describe it as one in any write-up. An optional expert
face-validity check on the scenarios themselves is in
`data/vignettes/vignette_validation_template.csv` (blank, for 2-3 raters to
fill in: is each scenario plausible, and is the assigned emotion label
appropriate).

**Experiment 3 — intervention message quality, two conditions.**
- *Oracle*: personalization is driven by the **gold** emotion label - an
  upper bound ("if detection were perfect, does personalization help?").
- *End-to-end*: personalization is driven by the **predicted** emotion from
  `llm_fewshot` (read from `results/exp2_predictions.csv`, so run Exp.2
  with `--use_llm` first). This shows how classification errors actually
  propagate into the final message, and reports a `need_match_rate` (how
  often a wrong emotion prediction still happens to map to the right SDT
  need - some confusions, like anger/disgust, share a need category and
  so don't hurt the downstream message).

Both conditions are scored against the canonical description of the
*gold* need (TF-IDF cosine similarity), with paired t-test / Wilcoxon
signed-rank tests. Also exports a **blind** A/B rubric
(`results/exp3_rubric_blind.csv` + a separate, rater-hidden
`exp3_rubric_key.csv`) for human expert scoring without raters knowing
which message is generic vs personalized - analyze filled-in copies with
`experiments/exp3b_rubric_analysis.py`. The automatic TF-IDF similarity
score is a cheap proxy and is structurally biased toward the personalized
templates (they share vocabulary with the need descriptions by
construction) - **the blind human rubric, not the automatic score alone,
is the result to lead with in a paper.**

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

## 8. Results (full run, no subsampling; `llm_fewshot` from a real `--use_llm` run)

### 8.1 A methodological fix that changes the conclusion — read this first

The original comparison trained `tfidf_logreg` on all 7 GoEmotions classes
and then evaluated it on ISEAR/vignettes, which only have 5 classes. That
classifier can still output "neutral"/"surprise" on those evaluations -
predictions that are *guaranteed wrong* and, worse, are invisible in a
same-label confusion matrix (they just don't show up anywhere, so the
matrix rows silently sum to less than the true class count). Checking this
directly: **46.5% of `tfidf_logreg`'s predictions on ISEAR, and 49.3% on
the vignettes, land on "neutral"/"surprise"** - i.e. nearly half of its
apparent error rate cross-domain was a label-space mismatch artifact, not
evidence that supervised ML generalizes worse than a word-counting
lexicon. `tfidf_logreg_5class` (identical algorithm, trained only on the 5
classes the evaluation actually uses) fixes this and is the fair
comparison point against `nrc_lexicon` and `llm_fewshot` (both of which
were always restricted to the eval label space by construction). **This
flips part of the original narrative**: properly compared, plain
supervised TF-IDF generalizes *at least as well as*, not worse than, the
lexicon-counting approach. Keep `tfidf_logreg` (7-class) in the table too -
it is a realistic "what happens if you just deploy your original model
into a narrower context" baseline - but do not call it a fair comparison
to the other two.

### 8.2 Classifier accuracy across the domain-shift ladder

| model | majority-class baseline | GoEmotions (in-domain) | ISEAR (cross-dataset) | Vignettes (near-domain) |
|---|:---:|:---:|:---:|:---:|
| (always predict the most frequent class) | 0.350 / 0.200 / 0.200 | — | — | — |
| `nrc_lexicon` | — | 0.384 | 0.326 | 0.360 |
| `tfidf_logreg` (7-class, **not** a fair comparison) | — | 0.599 | 0.259 (46.5% out-of-label-space) | 0.227 (49.3% out-of-label-space) |
| `tfidf_logreg_5class` (7-class trained, fair) | — | n/a | **0.381** | **0.387** |
| `llm_fewshot` (5 examples/class, n=150/class·n=750·n=75) | — | 0.467 | **0.785** | **0.893** |

See `results/figures/domain_shift_ladder.png` for this as a chart.

Three findings, not one:
1. **The lexicon-vs-TF-IDF story was partly an artifact** (8.1) - once
   fairly compared, classical supervised ML (`tfidf_logreg_5class`) is the
   best non-LLM option cross-domain, not the worst.
2. **`llm_fewshot` is in a different league on out-of-domain text**
   (0.785 ISEAR, 0.893 vignettes) even with only 5 in-context examples per
   class and no fine-tuning, confirming the original hypothesis that
   contextual/semantic understanding - not bigger classical models - is
   what actually closes the domain gap.
3. `llm_fewshot`'s **in-domain** GoEmotions score (0.467) is still its
   *lowest* of the three points, below `tfidf_logreg`'s in-domain score
   (0.599). GoEmotions' short, decontextualized, sarcasm-prone Reddit
   comments appear to be the hardest text register for all four methods,
   few-shot prompting included; see the confusion matrices in
   `results/figures/llm_fewshot_in_domain_cm.png` for where it struggles
   (`disgust`/`neutral`/`surprise` get confused with each other most).

The faculty-vignette result (0.893) should be read as closer to a **best
case** than a realistic field estimate: the vignettes were written by the
research team to be reasonably clear-cut illustrations of one emotion each
(see 8.4 on `disgust`), which is easier than naturally occurring,
ambiguous real text. ISEAR's 0.785 is probably the more representative
estimate of what to expect on natural (if not faculty-specific) text.

### 8.3 Per-emotion patterns worth reporting explicitly

Sadness (relatedness/isolation) remains the hardest emotion for the
non-LLM methods on the vignettes (`nrc_lexicon` 13%, `tfidf_logreg` 27%),
consistent with it being expressed through quiet, context-dependent
language rather than explicit "sad" vocabulary - `llm_fewshot` gets this
right 100% of the time, which is itself informative: this looks like a
case where contextual reasoning, not keyword overlap, is doing the work.

`disgust` is `llm_fewshot`'s weakest vignette category (60-88% depending
on the run) and is the one place classification noise reaches Experiment 3
(see 8.5). Looking at exactly which `disgust` vignettes get confused with
`anger` (`results/exp2_predictions.csv`), most are about moral/ethical
disapproval (favoritism, authorship unfairness, conflicts of interest)
rather than visceral disgust - a known hard boundary in emotion theory
(moral disgust vs. anger/moral outrage), not an implementation bug.

### 8.4 Intervention message quality (Experiment 3)

Oracle condition (personalization driven by the gold emotion label), TF-IDF
similarity to the target need description, n = 75:

| message type | mean similarity |
|---|:---:|
| generic (baseline) | 0.023 |
| personalized (oracle emotion) | 0.055 |

paired t-test p < 0.0001; Wilcoxon p < 0.0001. See
`results/figures/exp3_generic_vs_oracle.png`.

**Caveat that matters more than the p-value**: TF-IDF similarity to a need
description is a weak, automatable proxy that is structurally biased
toward the personalized templates (they share vocabulary with the need
descriptions by construction - a generic message could never score well
on this metric even if a human found it equally appropriate). Treat this
as a sanity check that the routing logic does what it says, not as
evidence of real motivational benefit. **The blind human rubric
(`results/exp3_rubric_blind.csv`, analyzed with
`experiments/exp3b_rubric_analysis.py`) is the result a paper should lead
with**; this repo ships the rubric and analysis code, but the actual human
ratings still need to be collected (3-5 raters, ~10-15 minutes each).

### 8.5 End-to-end condition (run Exp.2 with `--use_llm` first to enable)

Once `results/exp2_predictions.csv` contains `llm_fewshot` predictions,
Experiment 3 automatically also reports an **end-to-end** condition:
messages generated from the *predicted* (not gold) emotion, still scored
against the *true* underlying need. This is the number that actually
demonstrates the full pipeline, not just the personalization layer in
isolation. It also reports `need_match_rate`: how often a wrong emotion
prediction still happens to map to the correct SDT need (e.g. `anger` and
`disgust` both map to `autonomy`, so confusing them doesn't necessarily
hurt the message). Run it and report both the oracle-vs-end-to-end gap and
end-to-end-vs-generic significance - if personalization still beats
generic end-to-end despite real classifier noise, that is a much stronger
claim than the oracle number alone.
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
2. **LLM-based few-shot classification** (`--use_llm` in this repo) —
   **confirmed, not just hypothesized**: 0.785 cross-dataset / 0.893
   vignette accuracy with only 5 in-context examples per class (8.2),
   well above either non-LLM baseline. Because it draws on broad
   semantic/contextual knowledge rather than surface keyword overlap, it
   handles *implied* emotion (e.g. quiet isolation phrased without any
   "sad" word) far better than the other methods (8.3).
3. **A fine-tuned transformer** (`--use_transformer`). Likely improves the
   in-domain GoEmotions number somewhat over `tfidf_logreg` (published
   GoEmotions benchmarks with transformers are meaningfully but not
   dramatically higher), and may generalize a bit better cross-domain than
   raw TF-IDF n-grams because it carries pretrained semantic
   representations rather than purely lexical ones — but it is not
   expected to close the domain gap on its own, and is the least-tested
   option in this repo (no GPU/internet access in the sandbox this was
   built in - run it yourself to check).

None of these change the central methodological point: a classifier's
in-domain accuracy is not a safe proxy for how it will perform on a
different population, and that gap should be measured and reported, not
assumed away by picking a bigger model.

## 10. Limitations & future work

- No real faculty data was used or collected; Experiments 2-3 use a
  researcher-authored scenario set, not field data. An optional expert
  face-validity check on the scenarios is in
  `data/vignettes/vignette_validation_template.csv`, but it has not been
  filled in by independent raters yet - do that before treating the
  scenario set as validated.
- The emotion→need mapping (`src/sdt_mapping.py`) is a literature-informed
  proposal, not an empirically validated instrument — it should be reviewed
  by domain experts (e.g. organisational psychologists) before being used
  with real people.
- The automatic relevance metric (TF-IDF similarity) is a proxy for message
  quality and is structurally biased toward the personalized condition
  (8.4) - not a measure of actual motivational impact, and not strong
  enough to lead a paper's claims on its own. The blind rubric
  (`results/exp3_rubric_blind.csv`) is built and ready, but still needs
  3-5 independent human ratings collected and run through
  `experiments/exp3b_rubric_analysis.py` before publication.
- The end-to-end Experiment 3 condition (8.5) depends on having run
  Experiment 2 with `--use_llm` first; without that, only the oracle
  (upper-bound) condition is available, and that distinction must be kept
  explicit in any write-up - oracle results alone do not demonstrate the
  full pipeline.
- `disgust` remains a weak point even for `llm_fewshot` (8.3), plausibly
  because some of the constructed vignettes sit closer to moral
  disapproval than to visceral disgust - a known hard boundary in emotion
  theory, worth flagging rather than hiding.
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
