for ((i = 0 ; i < 5 ; i++ )); do python benchmark.py --batchsize 1024 --batches 100 aisio --datadir "train"; done
for ((i = 0 ; i < 5 ; i++ )); do echo 3 > /proc/sys/vm/drop_caches; python benchmark.py --batchsize 1024 --batches 100 dali --datadir "/data/train"; done
