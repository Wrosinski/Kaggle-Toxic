file=run_training.py
batch_size=256
kfold_run=1
importance=0
gpu=1
save_oof=1
save_models=0
load_models=1
output_submission=0
optimizer='Nadam'
data_type='BasicClean2'
run_name='KF3'


# LSTMdeep, LSTMconcat, LSTMconcat2, LSTMHierarchical

# python $file --model_name LSTMconcat2 --optimizer $optimizer \
#  --batch_size $batch_size --data_type $data_type \
#  --kfold_run $kfold_run --output_submission $output_submission \
#  --save_oof $save_oof --gpu $gpu --save_models $save_models \
#  --load_models $load_models --importance $importance \
#  --run_name $run_name &&

# python $file --model_name Conv2Dmodel --optimizer $optimizer \
#  --batch_size 256 --data_type $data_type \
#  --kfold_run $kfold_run --output_submission $output_submission \
#  --save_oof $save_oof --gpu $gpu --save_models $save_models \
#  --load_models $load_models --importance $importance \
#  --run_name $run_name &&

# python $file --model_name LSTMdeep2 --optimizer $optimizer \
#  --batch_size $batch_size --data_type $data_type \
#  --kfold_run $kfold_run --output_submission $output_submission \
#  --save_oof $save_oof --gpu $gpu --save_models $save_models \
#  --load_models $load_models --importance $importance \
#  --run_name $run_name &&

python $file --model_name GRUConvdeep3 --optimizer $optimizer \
  --batch_size $batch_size --data_type $data_type \
  --kfold_run $kfold_run --output_submission $output_submission \
  --save_oof $save_oof --gpu $gpu --save_models $save_models \
  --load_models $load_models --importance $importance \
  --run_name $run_name

# python $file --model_name LSTMconcat --optimizer $optimizer \
#  --batch_size $batch_size --data_type $data_type \
#  --kfold_run $kfold_run --output_submission $output_submission \
#  --save_oof $save_oof --gpu $gpu --save_models $save_models \
#  --load_models $load_models --importance $importance \
#  --run_name $run_name
