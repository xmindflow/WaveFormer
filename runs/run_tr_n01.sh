cd path/to/WaveFormer

# root_path
rp="/cabinet/dataset/Synapse/train_npz" # root dir for data

# test_path
tp="/cabinet/dataset/Synapse/test_vol_h5" # root dir for test

# output-dir
od="./model-out"

# num_workers
nw=16

# dstr_fast
dstrf=true

# max_epochs
epochs=1200

# max_iterations
mitr=300000

# optimizer
op="SGD"

# compact
compact=ture




pip install -r ./requirements.txt

python ./train.py --root_path ${rp} --test_path $tp --output_dir $od --continue_tr true --compact $compact --optimizer $op --max_epochs $epochs --max_iterations $mitr --num_workers $nw --dstr_fast $dstrf --en_lnum 1 --br_lnum 1 --de_lnum 1 --base_lr 0.05 --eval_interval 20 --batch_size 32 --model_name WaveFormer_n01


