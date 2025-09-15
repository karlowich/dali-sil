import sys
import time

import argparse
import numpy as np 
import cupy as cp
from nvidia.dali import pipeline_def, Pipeline
import nvidia.dali.types as types
import nvidia.dali.fn as fn
import sil

IMAGENET_MAX_SIZE = 15737107 # max size of images in the imagenet dataset
GDS_QD = 63
NLB = 7

@pipeline_def
def dali_pipe(data_dir):
    jpegs, labels = fn.readers.file(file_root=data_dir,
            shard_id=0,
            num_shards=1,
            random_shuffle=True,
            pad_last_batch=True,
            name="FILE"
        )

    return jpegs.gpu(), labels

class SILInputIterator(object):
    def __init__(self, device, data_dir, batch_size, backend, mnt="", gpu_nqueues=6, queue_depth=1024):
        self.n = sil.init(device, data_dir = data_dir, backend = backend, batch_size = batch_size, gpu_nqueues = gpu_nqueues, queue_depth = queue_depth, mnt = mnt)
        self.batch_size = batch_size
        
    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.n:
            self.__iter__()
            raise StopIteration

        labels = np.zeros((self.batch_size,), dtype = np.int32)
        batch = []
        arr = sil.next()
        for i in range(len(arr)):
            batch.append(cp.ndarray(shape=arr[i].shape, dtype=arr[i].dtype, memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(arr[i].ctypes.data, len(arr[i]), self), 0)))

        self.i += self.batch_size
        
        return batch, labels

    def __len__(self):
        return self.n
    
    def __del__(self):
        sil.term()

    next = __next__

@pipeline_def
def aisio_pipe(data_dir):
    pipe = Pipeline.current()
    batch_size = pipe.max_batch_size

    jpegs, labels = fn.external_source(
        source=SILInputIterator(device="/dev/libnvm0", backend = "libnvm-gpu", data_dir=data_dir, batch_size=batch_size), num_outputs=2, dtype=[types.UINT8, types.INT32]
    )
    
    return jpegs, labels




def setup():
    """Parse command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Benchmark loading data with DALI and xNVMe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batchsize", required=True, help='The number of samples to files per batch', type=int)
    parser.add_argument("--batches", required=False, help='The number of batches to load, default loads the entire dataset', type=int)
    subparsers = parser.add_subparsers(dest='dataloader', required=True)
    # Subcommand 'add'
    parser_dali = subparsers.add_parser('dali', help='Use DALI file reader for loading data')
    parser_dali.add_argument(
        "--datadir", help="The directory from which to load files", required=True
    )

    parser_python = subparsers.add_parser('aisio', help='Use AiSIO reader for loading data')
    parser_python.add_argument(
        "--datadir", help="The directory from which to load files", required=True
    )

    args = parser.parse_args()

    return args


def main():    

    args = setup()
    pipe = None
    if args.dataloader == "dali":
        pipe = dali_pipe(data_dir=args.datadir, batch_size=args.batchsize, num_threads=1, device_id=0)
        
    elif args.dataloader == "aisio":
        pipe = aisio_pipe(data_dir=args.datadir, batch_size=args.batchsize, num_threads=1, device_id=0)


    print(f"dataloader: {args.dataloader} batches: {args.batches} batchsize: {args.batchsize}")

    pipe.build()

    batches = args.batches

    if not batches:
        # Either run forever and wait for exception AiSIO or set limit (DALI)
        batches = sys.maxsize
        if args.dataloader == "dali":
            limit = pipe.epoch_size("FILE")
            batches = limit // args.batchsize
            if (limit % args.batchsize != 0):
                # if there is a remainder do one more iteration
                batches += 1

    # Warmup
    for _ in range(10):
        _ = pipe.run()

    mean_start = time.time()
    start = time.time()
    n = 0
    for i in range(batches):
        try:
            jpegs, _ = pipe.run()
        except StopIteration:
            break
        end = time.time()
        if (end - start) > 1:
            print(f"img/s: {(args.batchsize*(i - n))/(end - start)}") 
            n = i
            start = time.time()


    mean_end = time.time()
    print(f"Running Time: {mean_end - mean_start}") 
    print(f"Mean img/s: {(args.batchsize*batches)/(mean_end - mean_start)}") 
    exit(0)

if __name__ == "__main__":
    sys.exit(main())    
