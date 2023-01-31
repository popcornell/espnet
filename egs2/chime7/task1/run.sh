#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=kaldi/train_all
valid_set=kaldi/dev_gss_all
test_sets="kaldi/chime6/dev_gss kaldi/dipco/dev_gss kaldi/mixer6/dev_gss"

bpe_nlsyms=""
asr_config=conf/tuning/train_asr_transformer_wavlm_lr1e-4_specaugm_accum1_preenc128_warmup40k.yaml
inference_config="conf/decode_asr_transformer.yaml"
lm_config="conf/train_lm.yaml"
use_lm=false
use_word_lm=false
word_vocab_size=65000

./asr.sh \
    --lang en \
    --ngpu 4 \
    --token_type bpe \
    --nbpe 500 \
    --bpe_nlsyms "${bpe_nlsyms}" \
    --nlsyms_txt "data/nlsyms.txt" \
    --feats_type raw \
    --audio_format "flac" \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --use_lm ${use_lm} \
    --lm_config "${lm_config}" \
    --use_word_lm ${use_word_lm} \
    --word_vocab_size ${word_vocab_size} \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text" "$@"