import struct
import caffe
import lmdb
import os
import sys, getopt
import caffe.proto.caffe_pb2
from caffe.io import datum_to_array

def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print 'test.py -i <inputfile> -o <outputfile>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print 'test.py -i <inputfile> -o <outputfile>'
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
   print 'Input file is "', inputfile
   print 'Output file is "', outputfile

   lmdb_env = lmdb.open(inputfile)
   lmdb_txn = lmdb_env.begin()
   lmdb_cursor = lmdb_txn.cursor()
   datum = caffe.proto.caffe_pb2.Datum()

   num_datum = 0

   with open(outputfile,'wb') as f:
       for key, value in lmdb_cursor:
           datum.ParseFromString(value)
           #label = datum.label
           data = caffe.io.datum_to_array(datum)
           num_datum = num_datum + 1
           # print(num_datum)
           feat_dim_pt = 0
           f.write(struct.pack('f'*len(data),*data))
           if (num_datum % 1000) == 0:
               print(num_datum)
   f.closed


if __name__ == "__main__":
   main(sys.argv[1:])
            
