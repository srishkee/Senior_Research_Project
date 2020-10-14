import caffe
import numpy as np
import sys

# arg 0: system path (don't need to include)
# arg 1: /scratch/user/skumar55/mohanty/data/final_dataset/lmdb/grayscale-80-20/mean.binaryproto
# arg 2: train_mean.npy

if len(sys.argv) != 3:
	print("Usage: python convert_protomean.py proto.mean out.npy")
	sys.exit()

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( sys.argv[1] , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
out = arr[0]
np.save( sys.argv[2] , out )
print("Created mean file! Yay!")