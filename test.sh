python ltp_test.py \
	--train_file "./Dataset/train_demo.csv"\
	--validation_file "./Dataset/test_demo.csv"\
 	--task_name code \
	--per_device_train_batch_size 4 \
	--per_device_eval_batch_size 4 \
	--max_length 1024 \
 	--pad_to_max_length \
 	--model_name_or_path "checkpoints_10000/base/CODE/absolute_threshold/rate_0.01/temperautre_1e-05/lambda_0.1/lr_2e-05" \
 	--temperature  1e-5 \
 	--masking_mode hard

#       --task_name code

# python ltp_extract_and_save.py \
#	--train_file "./Dataset/train_10000.csv" \
#	--validation_file "./Dataset/test_10000.csv" \
#	--max_length 1024 \
#	--pad_to_max_length \
#	--model_name_or_path "checkpoints_10000/base/CODE/absolute_threshold/rate_0.01/temperautre_1e-05/lambda_0.1/lr_2e-05" \
#	--temperature  1e-5 \
#	--masking_mode hard \
#	"checkpoints_10000/base/CODE/absolute_threshold/rate_0.01/temperautre_1e-05/lambda_0.1/lr_2e-05/pruned_data"
	
# /hard/lr_2e-5" \
