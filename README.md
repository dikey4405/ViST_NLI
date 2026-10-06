# MoE_NLI_Multimodal
Vietnamese Text-Speech Multimodal Natural Language Inference

## Pipeline boundaries

Both feature pipelines save the same per-sample interface: four ordered
`4096`-dimensional pair features for `text_text`, `text_speech`, `speech_text`,
and `speech_speech`. They do not train an MoE classifier.

`phobert_xlsr_pipeline/train_alignment.py` trains only the modality-specific
text and speech projectors used before pair-feature extraction. It does not
replace or duplicate the MoE trainer.

All saved-feature model training is implemented in `source/training`. A run
loads exactly one configured feature source and never loads SONAR, PhoBERT, or
XLS-R encoders.

## Shared training commands

Run commands from the directory containing the `KLTN` package.

```powershell
python -m KLTN.source.training.train_conventional_moe --feature-source sonar
python -m KLTN.source.training.train_conventional_moe --feature-source phobert_xlsr
```

The same `--feature-source` option is available for `train_fine_grained_moe`
and `train_deepseek_moe`. The generic entry point also accepts any model config:

```powershell
python -m KLTN.source.training.train `
  --config conventional_moe.yaml `
  --feature-source phobert_xlsr
```

Checkpoints and logs are separated by both model and feature source, for
example `outputs/conventional_moe_counterfactual_utility/sonar` and
`outputs/conventional_moe_counterfactual_utility/phobert_xlsr`.

## Dense FFN baselines

Three controlled non-MoE baselines reuse the same saved feature files and
training entry point:

```powershell
python -m KLTN.source.training.train --config dense_ffn_mean.yaml --feature-source sonar
python -m KLTN.source.training.train --config dense_ffn_attention.yaml --feature-source sonar
python -m KLTN.source.training.train --config dense_ffn_full.yaml --feature-source sonar
```

Replace `sonar` with `phobert_xlsr` to run the other three experiments. Each
baseline applies one shared `FeedForwardExpert` to all four modes through the
same residual and `LayerNorm` structure used around routed experts.

`DenseFFN-Mean` uses arithmetic mean pooling and final cross entropy only.
`DenseFFN-Attention` adds semantic `ModeAttentionPooling`, still with final
cross entropy only. `DenseFFN-Full` additionally retains the shared mode head,
entropy reliability, reliability-aware attention, mode/invariance losses, and
the separate relation branch. None of the dense baselines creates a router,
top-k selection, expert list, shared routed expert, or balancing loss.

## Data and cache integrity

The default raw dataset is `KLTN/Dataset/{train,dev,test}.json`. Each split
uses `<split>_audio/premise_audio` and `<split>_audio/hypothesis_audio`.
The shared data loader also supports the older split-local
`Premise`/`Hypothesis` folders and explicit per-record audio paths.

PhoBERT/XLS-R manifests record a fingerprint of the source JSON and encoder
preprocessing settings. Speech fingerprints include WAV contents; text
fingerprints include audio availability because all four inputs are required.
Fingerprints are checked at resume and when loading caches for alignment or
extraction. This requires reading the audio files, but does not run an encoder.
An old manifest without a fingerprint must be rebuilt explicitly with
`cache.overwrite: true`; mismatched caches are never silently reused.

Alignment identifies sentences by normalized text, independently of NLI ids
and labels. Equal sentences in a batch are multiple positives in symmetric
InfoNCE and retrieval metrics. The relation loss in MoE training remains a
separate label-aware premise/hypothesis objective.

## Training and evaluation

The input projection produces `h_pre [B, 4, D]`. A shared `ModeNLIHead` predicts
three-class evidence for every mode, and normalized prediction entropy produces
a reliability value in `[0, 1]`. This reliability represents evidence
trustworthiness and conditions final modality fusion. It is not an input to the
proposed expert router.

The counterfactual utility router predicts expert suitability from `h_pre` and
learned mode identity. Its logits estimate the expected reduction in final NLI
cross entropy from forcing one mode through each candidate expert. Detached
targets use `U = L_base - L_counterfactual`; positive utility means that the
candidate expert improves the final decision. Target generation runs only for
labeled training/evaluation diagnostics. Standard inference remains a sparse
top-k forward without labels or counterfactual enumeration.

For MoE models, each routed expert runs only on its assigned rows. The routed output is added
to `h_pre` through a residual connection and normalized; DeepSeek also adds its
always-on shared-expert output. Learnable semantic attention then pools the four
ordered mode representations. When enabled, a small reliability score branch is
added to the attention score; reliability is detached on this path by default.

Classification pair features are always rebuilt from the original premise and
hypothesis embeddings. `PairAlignmentProjection` is an auxiliary head used only
by the label-aware relation contrastive loss, so it does not transform the
classification representation.

The shared evidence head is supervised by the original NLI label for all four
modes. Cross-mode invariance is the mean Jensen-Shannon divergence over the six
unordered mode pairs. Optional reliability weighting uses detached pair weights.
The complete training objective is:

```text
L_total = L_cls
        + lambda_balance * L_balance
        + lambda_relation * L_relation
        + lambda_mode * L_mode
        + lambda_invariance * L_invariance
        + lambda_cf * L_counterfactual_routing
```

The relation loss uses separate sums and counts for entailment and contradiction
pairs; neutral pairs do not contribute. `PairAlignmentProjection` remains an
auxiliary relation-only branch over the original premise and hypothesis vectors.
It never changes mode evidence, routing, attention, or final classification.

Epoch metrics are accumulated over the complete split. Reports include final
accuracy, macro/weighted F1, per-mode accuracy, reliability mean/std/median,
attention mean, row-normalized expert usage by mode, and router entropy by mode.
Utility runs additionally report signed target mean/standard deviation and
router top-1 agreement with the target-best expert. Router balancing remains a
sample-weighted mean because it is batch-dependent.

Training selects `best_model.pt` by maximizing dev macro-F1. The total loss
remains the optimization objective and is still reported for every split. On completion the shared
trainer restores that checkpoint, evaluates `test.pt` without gradient updates,
and writes `test_metrics.json` next to the checkpoint. `train.log` groups the
available main, auxiliary, mode, fusion, and routing metrics by epoch and split.
`metrics_history.jsonl` stores one complete machine-readable record after every
epoch so interrupted and consecutive runs retain their history. Both log files
use append mode, and a unique `run_id` identifies each training invocation.
The test report includes the selected epoch, selection metric and score, feature
source, and number of test samples. Test results do not participate in checkpoint
selection or early stopping.

Run the core tests from the directory containing `KLTN`:

```powershell
python -m unittest discover -s KLTN/source/tests -v
```

The retained tests cover SONAR features, PhoBERT/XLS-R extraction, dense and
MoE architectures, and the NLI auxiliary losses. They use synthetic data and
do not download models. These tests are independent of the runtime entry points.
