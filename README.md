## Introduction

This repository contains code for reproducing the unlearning final year project.

## Quick Start

### Prepare the environment
    pip install -r requirements.txt

### Run the scripts
    bash script/prune_ul.sh

    bash script/evaluate.sh

You can modify the parameter in the script:
such as the portion of forget set (e.g. forget01, forget05, retain99, retain05), the model saving path, and the unlearning threshold `unlearning_threshold` in `script/prune_ul.sh`


## Evaluation on TOFU

Clone the repo https://github.com/locuslab/open-unlearning and rename it as `tofu`. 
The folder structure should be like:

    |____unlearning_prune/
    |____tofu/

Follow the instructions on https://github.com/locuslab/open-unlearning to build the environment and prepare the data.

Use the following script to run the evalution:

    python src/eval.py --config-name=eval.yaml \
      experiment=eval/tofu/default \
      model=Llama-2-7b-chat-hf \
      model.model_args.pretrained_model_name_or_path='/root/autodl-tmp/unlearning_prune/${merged_output_model}' \
      task_name=SAMPLE_EVAL \
      retain_logs_path='/root/autodl-tmp/tofu/saves/eval/tofu_Llama-2-7b-chat-hf_retain${portion_of_retain}/TOFU_EVAL.json'

Make sure that `merged_output_model` matches the merged model weight you set in `script/evaluate.sh`, and `portion_of_retain` matches the `script/prune_ul.sh`

The output will be saved in `/tofu/saves/eval/SAMPLE_EVAL/TOFU_SUMMARY.json`
