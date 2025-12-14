#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=16000

train_set=tr_no_dev
valid_set=dev
test_sets="test"

train_config=conf/train_delay_smollm_1.7b_nods.yaml
inference_config=conf/decode_tts_espnet.yaml

bpe_opts="--subword_choice huggingface --subword_model HuggingFaceTB/SmolLM-1.7B"
codec_opts="--codec_choice ESPnet --codec_hf_model_tag ftshijt/espnet_codec_dac_large_v1.4_360epoch"
ssl_opts="--ssl_choice espnet_hubert --ssl_nlayer 18 --ssl_checkpoint_path exp/kmeans/38epoch.pth --ssl_kmeans_path exp/kmeans/xeus_18_5000clusters/km_5000.mdl --ssl_batch_bins 5000000"


# NOTE(yiwen) for NAACL 
# pretrained model + task specific tokens
# token_list_dir="data/token_list/svs_finetune_tts_vocab"
# tag="naacl_demo_1.7B_ext_phone_finetune_svs"


# NOTE(yiwen) for OPUSLM
# pretrained model + task specific tokens
token_list_dir="data/token_list/svs_finetune_tts_vocab"
tag="opuslm_v1_1.7B_anneal_ext_phone_finetune_svs_reproduce"


#"naacl_demo_1.7B_lora"
#"naacl_demo_1.7B_lr5e-6"
#"naacl_demo_1.7B_ext_phone_finetune_svs"

nj=1
inference_nj=2

./speechlm.sh \
    --task "svs" \
    --ngpu 1 \
    --nj "${nj}" \
    --inference_nj "${inference_nj}" \
    --data_name acesinger \
    --fs "${fs}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --token_list_dir "${token_list_dir}" \
    --tag "${tag}" \
    --audio_format "wav" \
    --min_wav_duration 3.0 \
    --max_wav_duration 30.0 \
    --nbest 10 \
    --gpu_inference true \
    ${codec_opts} \
    ${ssl_opts} \
    "$@"