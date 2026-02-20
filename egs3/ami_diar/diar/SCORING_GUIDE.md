# Diarization Scoring Guide

## Overview

This recipe uses **pyannote.metrics** for diarization evaluation, computing both:
- **DER (Diarization Error Rate)**: Standard metric with FA, Miss, and Speaker Error
- **JER (Jaccard Error Rate)**: Frame-level overlap metric

Both metrics are computed with **multiple collar settings** (0.0s, 0.25s, 0.5s) to analyze boundary sensitivity.

## Why pyannote.metrics?

✅ **Advantages**:
- Industry standard for diarization evaluation
- Well-maintained and widely used
- Provides detailed component breakdown (FA, Miss, Confusion)
- Supports both DER and JER
- Easy to integrate with pyannote.audio pipeline

❌ **Removed meeteval**:
- Not needed for standard diarization scoring
- Additional dependency
- DER computation already covered by pyannote

## Metrics Explained

### DER (Diarization Error Rate)

**Formula**: `DER = (FA + Miss + Confusion) / Total`

**Components**:
- **False Alarm (FA)**: Speech detected when there's no speech
- **Missed Detection (Miss)**: Speech not detected when there is speech
- **Speaker Confusion (Conf)**: Speech attributed to wrong speaker

**Lower is better**: 0% = perfect, 100% = completely wrong

### JER (Jaccard Error Rate)

**Formula**: `JER = 1 - Jaccard Index`

**Jaccard Index**: `|A ∩ B| / |A ∪ B|` (overlap / union)

**Interpretation**:
- Measures frame-level agreement between hypothesis and reference
- Accounts for both temporal and speaker assignment accuracy
- More sensitive to boundary errors than DER

**Lower is better**: 0% = perfect overlap, 100% = no overlap

### Collar

**Purpose**: Forgive boundary errors within a tolerance window

**Common values**:
- **0.0s (no collar)**: Strict evaluation, every boundary error counts
- **0.25s (250ms)**: Standard in literature (NIST RT evaluations)
- **0.5s (500ms)**: More forgiving, used for noisy/reverberant audio

**Effect on DER**:
- Larger collar → Lower DER (more forgiving)
- Collar mainly reduces FA and Miss components
- Speaker confusion mostly unaffected by collar

## Usage

### Option 1: Standalone Scoring Script

```bash
# Score with default collars (0.0, 0.25, 0.5)
python3 score.py \
    --hypothesis exp/debug_run/infer \
    --reference data/ami_debug/ground_truth

# Score with single collar
python3 score.py \
    --hypothesis exp/debug_run/infer \
    --reference data/ami_debug/ground_truth \
    --collar 0.25

# Score with custom collars
python3 score.py \
    --hypothesis exp/debug_run/infer \
    --reference data/ami_debug/ground_truth \
    --collar 0.0 0.1 0.25 0.5 1.0
```

### Option 2: Integrated with Inference

```bash
# Run inference and automatically score
python3 infer_and_score.py \
    --checkpoint exp/debug_run/checkpoints/last.ckpt \
    --data-dir data/ami_debug \
    --output-dir exp/debug_run/infer \
    --reference-dir data/ami_debug/ground_truth \
    --device cpu
```

### Option 3: Python API

```python
from score import compute_metrics_detailed

results = compute_metrics_detailed(
    hypothesis_dir="exp/debug_run/infer",
    reference_dir="data/ami_debug/ground_truth",
    collar_values=[0.0, 0.25, 0.5],
)

# Access results
der_025 = results[0.25]['der']
jer_025 = results[0.25]['jer']
print(f"DER (250ms collar): {der_025*100:.2f}%")
print(f"JER (250ms collar): {jer_025*100:.2f}%")
```

## Output Format

### Detailed Per-Collar Output

```
================================================================================
Evaluating: 250ms collar (collar=0.25s)
================================================================================
  ✓ IS1008a

--------------------------------------------------------------------------------
Results for 250ms collar:
--------------------------------------------------------------------------------
DER:            18.45%
  False Alarm:    3.12%
  Missed Speech:  8.23%
  Speaker Error:  7.10%

JER:            24.67%
--------------------------------------------------------------------------------
```

### Summary Table

```
================================================================================
Summary - Impact of Collar
================================================================================
Collar             DER      JER       FA     Miss  Spk Err
--------------------------------------------------------------------------------
No collar        22.34%   28.91%    4.23%   10.11%     8.00%
250ms collar     18.45%   24.67%    3.12%    8.23%     7.10%
500ms collar     15.78%   21.34%    2.45%    6.78%     6.55%
================================================================================
```

### Quick Summary

```
Quick Summary (250ms collar):
  DER=18.45%  JER=24.67%
```

## RTTM File Format

Both hypothesis and reference must be in RTTM format:

```
SPEAKER recording-id 1 start-time duration <NA> <NA> speaker-id <NA> <NA>
```

Example:
```
SPEAKER IS1008a 1 0.500 2.345 <NA> <NA> speaker1 <NA> <NA>
SPEAKER IS1008a 1 1.200 3.450 <NA> <NA> speaker2 <NA> <NA>
SPEAKER IS1008a 1 3.800 1.234 <NA> <NA> speaker1 <NA> <NA>
```

Fields:
- **recording-id**: Unique identifier for the recording
- **start-time**: Segment start in seconds
- **duration**: Segment duration in seconds
- **speaker-id**: Speaker label (arbitrary, e.g., speaker1, speaker2, spk_0)

## Expected Performance

### Debug Run (3 meetings, 3 epochs)
- **DER (no collar)**: 30-40%
- **DER (250ms collar)**: 25-35%
- **JER (250ms collar)**: 30-40%

### Full Training (135 meetings, 50 epochs)
- **WavLM + PIT**:
  - DER (no collar): 18-20%
  - DER (250ms collar): 15-17%
  - JER (250ms collar): 20-24%

- **XEUS + PIT**:
  - DER (no collar): 15-18%
  - DER (250ms collar): 12-15%
  - JER (250ms collar): 18-22%

### State-of-the-art on AMI (IHM)
- **Best published**: ~10-12% DER (250ms collar)
- Typically achieved with:
  - Large SSL models (WavLM Large or XEUS)
  - Speaker embeddings + clustering
  - Multi-stage training
  - Ensemble methods

## Interpreting Results

### DER Components

**High False Alarm (> 10%)**:
- Model is too aggressive
- Predicting speech where there's silence
- Solutions: Adjust binarization threshold, add voice activity detection

**High Missed Detection (> 10%)**:
- Model is too conservative
- Missing short speech segments
- Solutions: Lower binarization threshold, use smaller chunk size

**High Speaker Confusion (> 10%)**:
- Model confusing speaker identities
- Solutions: Add speaker embeddings, use PIT (already implemented), increase conformer depth

### JER vs DER

**JER > DER (typical)**:
- JER is stricter about boundaries
- JER penalizes partial overlaps more heavily

**Large JER-DER gap (> 10%)**:
- Poor temporal alignment (boundary errors)
- Solutions: Post-process with smoothing, median filtering (already implemented)

### Collar Impact

**Large collar effect (> 5% DER reduction)**:
- Boundary errors are significant
- Model predictions are temporally noisy
- Solutions: Use larger median filter, add boundary refinement

**Small collar effect (< 2% DER reduction)**:
- Boundaries are already accurate
- Errors are primarily FA, Miss, or Speaker Confusion

## Comparison with Other Metrics

### DER vs WER (Word Error Rate)
- **DER**: Who spoke when
- **WER**: What was said
- Both use similar error components (insertion, deletion, substitution)

### DER vs cpWER (concatenated minimum permutation WER)
- **cpWER**: Used in multi-talker ASR
- **DER**: Used in speaker diarization
- cpWER includes recognition accuracy, DER only cares about speaker labels

### DER vs SDR (Speaker Diarization Rate)
- **SDR**: Sometimes used interchangeably with DER
- In some papers, SDR = 1 - DER (higher is better)
- We report DER (lower is better) following NIST convention

## Common Issues

### Issue: "No matching reference files found"
**Cause**: RTTM filenames don't match between hypothesis and reference
**Solution**: Ensure consistent naming (recording_id.rttm)

### Issue: DER > 100%
**Cause**: Massive false alarm (predicting much more speech than exists)
**Solution**: Check model output, adjust binarization, use VAD

### Issue: DER = 100%
**Cause**: Complete mismatch (all errors)
**Solution**: Check if hypothesis is empty, verify RTTM format, check speaker labels

### Issue: All speakers mapped to one speaker
**Cause**: Model not learning speaker discrimination
**Solution**: Verify PIT is enabled, check powerset encoding, increase model capacity

## Advanced Usage

### Custom Collar Values

```python
results = compute_metrics_detailed(
    hypothesis_dir="exp/output",
    reference_dir="data/reference",
    collar_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],  # Fine-grained analysis
)
```

### Per-File Results

```python
from pyannote.metrics.diarization import DiarizationErrorRate

metric = DiarizationErrorRate(collar=0.25)

for hyp_rttm, ref_rttm in zip(hypothesis_files, reference_files):
    hyp = load_rttm_as_annotation(hyp_rttm)
    ref = load_rttm_as_annotation(ref_rttm)

    file_der = metric(ref, hyp, detailed=True)
    print(f"{hyp_rttm.stem}: DER={file_der*100:.2f}%")
```

### Detailed Component Analysis

```python
metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)

# Process all files
for ref, hyp in zip(references, hypotheses):
    metric(ref, hyp)

# Get detailed breakdown from accumulated metrics
total_der = abs(metric)
components = {
    'false alarm': metric['false alarm'] / metric['total'],
    'missed detection': metric['missed detection'] / metric['total'],
    'confusion': metric['confusion'] / metric['total'],
}

print(f"False Alarm Rate:   {components['false alarm']*100:.2f}%")
print(f"Miss Rate:          {components['missed detection']*100:.2f}%")
print(f"Confusion Rate:     {components['confusion']*100:.2f}%")
print(f"Total (DER):        {total_der*100:.2f}%")
```

## References

1. **pyannote.metrics**: https://github.com/pyannote/pyannote-metrics
2. **NIST RT Evaluations**: https://www.nist.gov/itl/iad/mig/rich-transcription-evaluation
3. **DER Definition**: NIST RT-09 Evaluation Plan
4. **JER (Jaccard)**: Ryant et al. "The Second DIHARD Diarization Challenge" (2019)
5. **Collar Tolerance**: NIST standard - 250ms collar for RT evaluations

## Summary

**Key Points**:
- ✅ Uses pyannote.metrics (industry standard)
- ✅ Computes both DER and JER
- ✅ Multiple collar settings (0.0s, 0.25s, 0.5s)
- ✅ Detailed component breakdown
- ✅ Standalone scoring script + integrated pipeline
- ❌ Removed meeteval dependency

**Usage**:
```bash
# Quick scoring
python3 score.py --hypothesis exp/output --reference data/reference

# With inference
python3 infer_and_score.py --checkpoint exp/model.ckpt --reference-dir data/reference
```

**Expected Results** (AMI debug, 3 epochs):
- DER (250ms collar): 25-35%
- JER (250ms collar): 30-40%
