# ESPnet2 Speech Language Model Singing Voice Synthesis (SLMSVS) Recipe
🎙️ ``Task``: Singing Voice Synthesis (SVS)

📊 ``Corpus``: Mandarin Multiple Singers Corpus, [ACESinger (ACE-Opencpop)](https://arxiv.org/abs/2401.17619).

Before start working on this recipe, please refer to tutorials
- [PSC usage tutorial](https://www.wavlab.org/activities/2022/psc-usage/)
- [Espnet recipe tutorial](https://github.com/espnet/notebook/blob/master/ESPnet2/Course/CMU_SpeechRecognition_Fall2022/recipe_tutorial.ipynb)
- [Speechlm Template](https://github.com/espnet/espnet/blob/speechlm/egs2/TEMPLATE/speechlm1/README.md#stage-10-evaluation)

We use [VERSA](https://github.com/shinjiwlab/versa/) for evaluation.

## Data Structure
* Download ACE-Opencpop corpus from [this link](https://drive.google.com/file/d/1qHLW3U7a0z8FpWuaEUmY-LViBIwRmeM0/view?usp=drive_link)

* Organize the data as
    ```
    acesinger/speechlm1
        |__downloads
            |__ACESinger
                |__lyrics
                |__no_lyrics
                |__raw_data
                |__segments
        
    ```

* The SLMSVS template is organized as stated in [our paper](https://openreview.net/pdf?id=4rsd8OuIR0).
* More specifically, we use modified ``label`` as the conditional entry, ``wav.scp`` as the target entry. Information from ``text`` and ``score`` modalities of the original SVS are integrated to ``label`` through data preprocessing (stage 2). During processing, the tokens from SVS are distinguished by tokens from TTS using a ``svs_`` prefix.

## Run the Recipe

Use stage1-5 for data preparation and token list generation. Please make sure you don't have overlapping validation set and test set when running stage2, which will duplicatedly preprocess validation set and cause errors. Feel free to change the config in ``./run.sh`` after stage2.
```
./run.sh --stage 1 --stop_stage 5 
```

We fine-tune on a pretrained TTS model ([download and setup](https://github.com/espnet/espnet/blob/speechlm/egs2/TEMPLATE/speechlm1/README.md#pretrained-models)). 

To combine the svs tokens to the pretrained token list, run

```
python pyscripts/utils/speechlm_extend_vocab.py \
  --input_token_list_dir data/token_list/llm_vocab2 \
  --output_token_list_dir data/token_list/svs_finetune_tts_vocab \
  --input_exp_dir exp/speechlm_opuslm_v1_1.7B_anneal \
  --output_exp_dir exp/speechlm_opuslm_v1_1.7B_anneal_ext_phone_finetune_svs_reproduce \
  --model_name 117epoch.pth \
  --additional_vocabs dump/raw_svs_acesinger/tr_no_dev/token_lists/svs_lb_token_list \
  --additional_task svs
```

Skip stage 6 if you use the pretrained model and already conducted the above token extension.

```
./run.sh --stage 7
```

## Inference
```
./ds_inference.sh
```