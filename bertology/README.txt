CUDA_VISIBLE_DEVICES=0 nohup python $PWD/bertology/run_squad.py --model_type bert --model_name_or_path bert-base-uncased --do_train --do_eval --do_lower_case --train_file $PWD/data/bert-10/train.json --predict_file $PWD/data/bert-10/train.json --per_gpu_train_batch_size=32 --per_gpu_eval_batch_size=32 --output_dir $PWD/results/exp0 --evaluate_during_training > $PWD/logs0.txt &


Test
====

SAMPLE=bert-10; EXPNAME=0 ; rm -rf $PWD/results/$EXPNAME ; rm -rf $PWD/data/$SAMPLE/cached* ; CUDA_VISIBLE_DEVICES=0 python $PWD/bertology/run_squad.py --model_type memory --model_name_or_path bert-base-uncased --do_train --do_eval --do_lower_case --train_file $PWD/data/$SAMPLE/train.json --predict_file $PWD/data/$SAMPLE/dev.json --per_gpu_train_batch_size=8 --per_gpu_eval_batch_size=8 --output_dir $PWD/results/$EXPNAME --save_steps -1 --num_train_epochs 3

Eval
====

SAMPLE=bert-10; EXPNAME=0 ; CUDA_VISIBLE_DEVICES=0 python $PWD/bertology/run_squad.py --model_type memory --model_name_or_path bert-base-uncased --do_eval --do_lower_case --train_file $PWD/data/$SAMPLE/train.json --predict_file $PWD/data/$SAMPLE/dev.json --per_gpu_train_batch_size=8 --per_gpu_eval_batch_size=8 --output_dir $PWD/results/$EXPNAME
