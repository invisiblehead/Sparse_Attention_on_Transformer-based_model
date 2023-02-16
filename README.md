# Sparse_Attention_on_Transformer-based_model


#### This is the replication package and dataset for FSE'23 submission.
#### For the detailed repliation package, please go to: https://github.com/invisiblehead/Sparse_Attention_on_Transformer-based_model
For the processed dataset, please go to: https://zenodo.org/deposit/7606184 Also, the original dataset without preprocessing can be found here: https://osf.io/d45bw/


### How to replicate the experiment:

#### 1. Set up the virtual environments with anaconda and activate it
       $ conda create -n ltp python=3.8
       $ conda activate ltp
#### 2. Install the dependencies (Transformer, etc)
       $ pip install -r requirements.txt
       
### Replace the standard Transformer implementation files to support the functionality of token pruning based one sparsity attention. Here are files we implement and change for the implementation of Learned Token Pruning (LTP) based on the code base of Hugging Face (https://huggingface.co/) Transformer. All required files are under transformer_revised/:

#### Under `/anaconda/envs/ltp/lib/python3.8/site-packages/transformers/`
##### You may also use revise_local_transformer_package.sh to replace the files under `/anaconda/envs/ltp/lib/python3.8/site-packages/transformers/`. Please take care of the `src` and `des`.

* `trainer_ltp.py`: replaced the trainer.py with trainer_ltp.py in my implementation

* `trainer.py`: fix minor deprecated error

* `modeling_utils.py`: minor change to fix error

* `__init__.py`: update ltp 

* `training_args.py`: update the arguments of ltp(lr_threshold, lambda_threshold, weight_decay_threshold, masking_mode, temperature, etc.) to pass it to longformer 

* `models/`

  * `__init__.py`: update ltp
  
  * `ltp/`: major change on most of the files within this folder
  
    * `__init__.py`: replace the deprecated modules like _BaseLazyModule with _LazyModule
    * `configuration_ltp.py` 
    * `ltp_model_output.py`: add global attention and attention_mask
    * `modeling_ltp.py`: replace the IBERT with longformer, given LTP implemented based on IBERT, update the attention mechanism only for sequence classification task
    * `prune_modules.py`: implement on how to get average score of local and global attention for pruning in LTP
    
  * `auto/`: updated the ltp configurations given LTP is not clarified
    * `configuration_auto.py`: updated the ltp configurations
    * `modeling_auto.py`: updated the ltp configurations


## How to use the Token pruning with longformer:

### Once the conda virtual env: ltp1 is set up on your VM (Please don't update the default transformer in this virtual env, since it takes time and effort to set up ltp in huggingface transformer.):

#### 1. To activate the virtual env:
       $ conda activate ltp
#### 2. prepare for the checkpoint

You can also go to https://github.com/kssteven418/LTP/blob/ltp/main/README.md for the meaning of each argument.

This is just a demo about how to play with the code and run the whole process. Please use your own train_file and val_file, and update with your own output_dir. And update these arguments based the your task and the computing power of your GPU.
For security defect detection datasets, please specify the task name here:

    $ python ./run_glue_code_metrics.py --model_name_or_path allenai/longformer-base-4096 --do_train --do_eval --train_file ./Data/train_subset.csv --validation_file ./Data/test.csv --max_seq_length 1024 --per_device_train_batch_size 4 --learning_rate 2e-5 --num_train_epochs 3 --overwrite_output_dir --output_dir code_test



After generate the checkpoint, the final model will be checkpointed in cmd_test.
* Remove cmd_test/trainer_state.json
* In the configuration file cmd_test/config.json, change (1) "architectures" to ["LTPForSequenceClassification"] and (2) "model_type" to "ltp"
* Add the following lines in the configuration file cmd_test/config.json:  "prune_mode": "absolute_threshold", "final_token_threshold": 0.01,

Final_token_threshold determines the token threshold of the last layer, and the thresholds of the remaining layers will be linearly scaled. For instance, the thresholds for the 3rd, 6th, and 9th layers will be 0.0025, 0.005, and 0.0075, respectively, when setting the final_token_threshold , i.e., the threshold for the last (12th) layer, to 0.01. This number is a hyperparameter, and we found that 0.01 works well in many cases.

#### 3. Run the soft pruning (remember to delete the previously generated checkpoint in the checkpoints folder before you rerun the soft pruning for the same task with the same parameter in case that it won’t overwrite):
    $ python ./run_code_metrics.py --arch ltp-base --task CODE --restore code_test --train_file ./Data/train_subset.csv --validation_file ./Data/test.csv --lr 2e-5 --temperature 1e-5 --lambda 0.1 --weight_decay 0 --bs 4 --masking_mode soft --epoch 3 --save_step 100 --no_load

The final model will be checkpointed in {CKPT_soft} = checkpoints/base/CODE/absolute_threshold/rate_{final_token_threshold}/temperature_{T}/lambda_{lambda}/lr_{lr}. Remove trainer_state.json from the checkpoint file in {CKPT_soft}.

#### 4. Run the hard pruning:
     $ python ./run_code_metrics.py --arch ltp-base --task CODE --restore checkpoints/base/CODE/absolute_threshold/rate_0.03/temperautre_1e-05/lambda_0.1/lr_2e-05 --train_file ./Data/train_subset.csv --validation_file ./Data/test.csv --lr 2e-5 --bs 4 --masking_mode hard --epoch 1 --save_step 100 --eval

The final model will be checkpointed in {CKPT_soft}/hard/lr_{LR}

#### 5. Run the visualization(demo for text classification task). (Please make sure the pathes of train_file, validation file and model_name_or_path are correct):
      $ bash test.sh
Please see ltp_test.py for more information
   

#### 6. save the dataset for powershell after token pruning. (Please make sure the pathes of train_file, validation file, model_name_or_path and the output dir are correct):
      $ bash test.sh
 please see : ltp_extract_and_save.py for more information
