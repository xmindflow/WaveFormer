cd path/to/WaveFormer

# root_path
vp="/cabinet/dataset/Synapse" # root dir for data

# output-dir
od="./model-out"

# num_workers
nw=16

# dstr_fast
dstrf=true

# compact
compact=ture

wfp="${od}/WaveFormer_v8801/WaveFormer_v8801_epoch_1099.pth"


pip install -r ./requirements.txt

python ./test.py --volume_path $vp --output_dir $od --compact $compact --num_workers $nw --dstr_fast $dstrf --en_lnum 1 --br_lnum 1 --de_lnum 1 --batch_size 32 --weights_fpath $wfp


