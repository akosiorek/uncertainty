import os
import shutil
import random

import lmdb

import caffe.proto.caffe_pb2

LOG_EVERY = 10000


def extract_samples(input_db, output_db, sample_keys, shuffle=True):

    if os.path.exists(output_db):
        shutil.rmtree(output_db)

    map_size = 2**40

    env_in = lmdb.open(input_db, readonly=True, map_size=map_size)
    env_out = lmdb.open(output_db, readonly=False, map_size=map_size)

    sample_keys = sorted(sample_keys)
    ids = range(len(sample_keys))

    if shuffle:
        random.shuffle(ids)

    num = 0
    with env_in.begin() as txn_in:
        cursor_in = txn_in.cursor()
        with env_out.begin(write=True) as txn_out:
            for id, key in zip(ids, sample_keys):
                txn_out.put(str_id(id), cursor_in.get(key))

                num += 1
                if num % LOG_EVERY == 0:
                    print 'Stored {0:8} samples in {1}'.format(num, output_db)

    env_in.close()
    env_out.close()
    if num % LOG_EVERY != 0:
        print 'Stored {0:8} samples in {1}'.format(num, output_db)


def entry_shape(db_path):

    env = lmdb.open(db_path, readonly=True)
    with env.begin() as txn:
        cursor = txn.cursor()

        cursor.first()
        key, raw_datum = cursor.item()
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(raw_datum)

        shape = datum.channels, datum.height, datum.width
    env.close()
    return shape


def size(db_path):
    env = lmdb.open(db_path, readonly=True)
    size = 0
    with env.begin() as txn:
        for _ in txn.cursor():
            size += 1
    env.close()
    return size


def move_cursor_circular(cursor):
    moved = cursor.next()
    if not moved:
        cursor.first()


def str_id(num):
    return '{:08}'.format(num)