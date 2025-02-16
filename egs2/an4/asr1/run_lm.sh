#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

./asr.sh \
    --lang en \
    --asr_config conf/train_asr_transformer.yaml \
    --inference_config conf/decode_asr.yaml \
    --lm_config conf/train_lm.yaml \
    --train_set train_nodev \
    --valid_set train_dev \
    --test_sets "train_dev test" \
    --bpe_train_text "dump/raw/train_nodev/text" \
    --lm_train_text "data/train_nodev/text" "$@"
