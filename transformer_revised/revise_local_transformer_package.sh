#!/bin/bash
src=/content/drive/MyDrive/Longformer/transformers_revised/ 
des=/usr/local/envs/ltp/lib/python3.8/site-packages/transformers


for i in trainer_ltp.py trainer.py modeling_utils.py __init__.py training_args.py models/__init__.py models/ltp models/auto/configuration_auto.py models/auto/modeling_auto.py; do
    echo "cp -rT $src/$i $des/$i"
    cp -rT $src/$i $des/$i
done

