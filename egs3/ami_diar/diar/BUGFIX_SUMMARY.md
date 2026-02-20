# Bug Fixes Summary

## Issue Reported

User reported: **"the debug code is not really running"**

Training was completing instantly without processing any batches, showing `Epoch 2: 0%|...|0/200 [00:00<?, ?it/s]`.

## Root Causes Found

### Bug #1: Lazy Iterator Not Subscriptable (FIXED ✅)
**File**: `src/dataset.py` line 68-69
**Error**: `TypeError: 'LazyManifestIterator' object is not subscriptable`
**Cause**: `CutSet.from_file()` returns a lazy iterator, but line 215 tries to access cuts by ID: `cut = self.cuts[cut_id]`
**Fix**: Convert to eager loading:
```python
# Before:
self.cuts = CutSet.from_file(manifest_path)

# After:
self.cuts = CutSet.from_file(manifest_path).to_eager()
```

### Bug #2: Conformer Mask Creation Error (FIXED ✅)
**File**: `espnet3/components/diarization/segmentation_model.py` line 184-188
**Error**: `TypeError: int() argument must be a string, a bytes-like object or a real number, not 'list'`
**Cause**: Attempted to manually create masks for Conformer, but the lengths tensor wasn't properly formatted
**Fix**: Let ConformerEncoder create masks internally:
```python
# Before:
if lengths is not None:
    masks = (~make_pad_mask(lengths)).unsqueeze(1)  # (B, 1, T)
else:
    masks = None
encoded, _, _ = self.conformer(projected, masks)

# After:
# Conformer uses ilens internally, so pass lengths directly
# It will create masks inside the conformer.forward() method
encoded, _, _ = self.conformer(projected, ilens=lengths)
```

## Current Status

### ✅ Fixed Issues:
1. Dataset lazy iterator bug - cuts can now be accessed by ID
2. Conformer mask creation - now uses proper `ilens` parameter

### ⚠️ Remaining Issue:
Training process crashes during sanity checking with `exit code 134` (SIGABORT).
This suggests a low-level crash in the Conformer forward pass, possibly in MPS backend or attention computation.

### Potential Causes:
1. **MPS (Apple Silicon) incompatibility** with certain operations in Conformer
2. **Memory issue** during attention computation
3. **Numerical instability** in the model

## Next Steps to Debug

1. **Test with CPU**: Disable MPS to see if it's a backend issue
   ```python
   trainer = L.Trainer(
       accelerator="cpu",  # Force CPU
       ...
   )
   ```

2. **Add error handling**: Wrap forward pass in try-except to capture error
   ```python
   try:
       encoded, _, _ = self.conformer(projected, ilens=lengths)
   except Exception as e:
       print(f"Conformer error: {e}")
       print(f"  projected shape: {projected.shape}")
       print(f"  ilens: {lengths}")
       raise
   ```

3. **Test smaller batch**: Reduce batch size to 1 to rule out batch-related issues

4. **Check tensor dtypes**: Ensure all tensors have compatible dtypes (float32)

5. **Verify lengths format**: Ensure `lengths` is a 1D LongTensor, not nested

## Files Modified

1. **src/dataset.py**:
   - Line 69: Added `.to_eager()` to load cuts eagerly

2. **espnet3/components/diarization/segmentation_model.py**:
   - Lines 179-185: Removed manual mask creation, pass `ilens` to Conformer

3. **espnet3/components/diarization/ssl_frontend.py**:
   - Added full XEUS support (separate from these bugs)

## Testing Results

- ✅ Dataset loads correctly: 3 cuts, 1499 chunks
- ✅ Dataloader works: 749 train batches, 1 val batch
- ✅ Model initializes: 95.3M params (94.4M frozen, 902K trainable)
- ✅ Forward pass works in eval mode (without Lightning)
- ❌ Training crashes during Lightning sanity check

## Verification Commands

```bash
# Test dataset
python3 << 'EOF'
from src.dataset import DiarizationDataset
dataset = DiarizationDataset('data/ami_debug/train_cuts.jsonl.gz', 4.0, 3.0, 0.02, 4, 16000)
print(f"Chunks: {len(dataset.chunks)}")
print(f"First item: {dataset[0]['speech'].shape}")
EOF

# Test forward pass (without Lightning)
python3 << 'EOF'
import torch
from src.dataset import DiarizationDataset, collate_fn
from espnet3.components.diarization.segmentation_model import PowersetDiarizationModel
from torch.utils.data import DataLoader

dataset = DiarizationDataset('data/ami_debug/train_cuts.jsonl.gz', 4.0, 3.0, 0.02, 4, 16000)
loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
model = PowersetDiarizationModel(ssl_model_name="wavlm_base", ssl_freeze=True, ssl_num_layers=13,
                                 ssl_hidden_size=768, projection_size=128, conformer_num_blocks=2,
                                 num_speakers=4, max_speakers_per_frame=2)
model.eval()

batch = next(iter(loader))
with torch.no_grad():
    logits, out_lengths = model(batch["speech"], batch["speech_lengths"])
    loss, stats = model.compute_loss(logits, batch["labels"], out_lengths)
print(f"✓ Loss: {loss.item():.4f}, Accuracy: {stats['accuracy']:.4f}")
EOF
```

## Summary

**Two critical bugs were fixed**:
1. Lazy iterator subscripting bug in dataset
2. Conformer mask creation bug in segmentation model

**Training is still not functional** due to a low-level crash during Lightning's sanity check. This appears to be a deeper issue, possibly related to the MPS backend or numerical operations in the Conformer encoder.

**Recommendation**: Test with `accelerator="cpu"` to isolate whether this is an MPS-specific issue.
