file=run_training.py
batch_size=256
kfold_run=1
importance=0
gpu=0
save_oof=1
save_models=0
load_models=1
output_submission=1
optimizer='Adam'
data_type='BasicClean2'
run_name='KF1'


# GRUdeep, GRUconcat, GRUconcat2, GRUHierarchical

python $file --model_name LSTMDeepmoji --optimizer $optimizer \
  --batch_size $batch_size --data_type $data_type \
  --kfold_run $kfold_run --output_submission $output_submission \
  --save_oof $save_oof --gpu $gpu --save_models $save_models \
  --load_models $load_models --importance $importance \
  --run_name $run_name &&

python $file --model_name GRUdeep --optimizer $optimizer \
  --batch_size $batch_size --data_type $data_type \
  --kfold_run $kfold_run --output_submission $output_submission \
  --save_oof $save_oof --gpu $gpu --save_models $save_models \
  --load_models $load_models --importance $importance \
  --run_name $run_name &&

python $file --model_name GRUconcat2 --optimizer $optimizer \
  --batch_size $batch_size --data_type $data_type \
  --kfold_run $kfold_run --output_submission $output_submission \
  --save_oof $save_oof --gpu $gpu --save_models $save_models \
  --load_models $load_models --importance $importance \
 --run_name $run_name &&

python $file --model_name GRUconcat --optimizer $optimizer \
 --batch_size $batch_size --data_type $data_type \
 --kfold_run $kfold_run --output_submission $output_submission \
 --save_oof $save_oof --gpu $gpu --save_models $save_models \
 --load_models $load_models --importance $importance \
--run_name $run_name &&

python $file --model_name GRUHierarchical --optimizer $optimizer \
  --batch_size $batch_size --data_type $data_type \
  --kfold_run $kfold_run --output_submission $output_submission \
  --save_oof $save_oof --gpu $gpu --save_models $save_models \
  --load_models $load_models --importance $importance \
  --run_name $run_name
