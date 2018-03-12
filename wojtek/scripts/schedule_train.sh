file=run_training.py
batch_size=128
kfold_run=1
importance=0
gpu=0
save_oof=1
save_models=1
load_models=0
output_submission=0
optimizer='Adam'
data_type='BasicClean2'
run_name='KF2'


# LSTMdeep, LSTMconcat, LSTMconcat2, LSTMHierarchical

python $file --model_name LSTMconcat2 --optimizer $optimizer \
  --batch_size $batch_size --data_type $data_type \
  --kfold_run $kfold_run --output_submission $output_submission \
  --save_oof $save_oof --gpu $gpu --save_models $save_models \
  --load_models $load_models --importance $importance \
 --run_name $run_name &&

python $file --model_name LSTMdeep --optimizer $optimizer \
  --batch_size $batch_size --data_type $data_type \
  --kfold_run $kfold_run --output_submission $output_submission \
  --save_oof $save_oof --gpu $gpu --save_models $save_models \
  --load_models $load_models --importance $importance \
  --run_name $run_name &&

  python $file --model_name LSTMconcat --optimizer $optimizer \
   --batch_size $batch_size --data_type $data_type \
   --kfold_run $kfold_run --output_submission $output_submission \
   --save_oof $save_oof --gpu $gpu --save_models $save_models \
   --load_models $load_models --importance $importance \
  --run_name $run_name &&

python $file --model_name LSTMHierarchical --optimizer $optimizer \
  --batch_size 256 --data_type $data_type \
  --kfold_run $kfold_run --output_submission $output_submission \
  --save_oof $save_oof --gpu $gpu --save_models $save_models \
  --load_models $load_models --importance $importance \
  --run_name $run_name
