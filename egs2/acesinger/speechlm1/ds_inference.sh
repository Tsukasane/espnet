#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00

# naacl_demo_1.7B_lr5e-6
sampling_temperature=0.8
tag="opuslm_v1_1.7B_anneal_ext_phone_finetune_svs_reproduce"
inference_model=finetune_45epoch.pth

#"naacl_demo_1.7B_ext_phone_finetune_svs"

source path.sh

bash run.sh \
    --ngpu 1 \
    --nj 2 \
    --inference_nj 16 \
    --skip_data_prep false \
    --stage 9 --stop_stage 9 \
    --tag ${tag} \
    --inference_model ${inference_model} \
    --inference_config conf/decode_tts_espnet.yaml  \
    --inference_args "--sampling_temperature ${sampling_temperature}" \
    --task svs \
    --nbest 2