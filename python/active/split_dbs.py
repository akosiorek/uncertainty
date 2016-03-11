#!/usr/bin/env python
import sys
import random

import lmdb
import caffe.proto.caffe_pb2

import db


def keys_per_category(db_path):
    cats = {}

    env = lmdb.open(db_path, readonly=True)
    with env.begin() as txn:
        for key, raw_datum in txn.cursor():

            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(raw_datum)
            label = datum.label

            if label not in cats:
                cats[label] = []
            cats[label].append(key)
    env.close()
    return cats


def split_dbs(input_db, output_dbs, percent_in_first, shuffle=True):

    map_size = 2**40

    keys_per_cat = keys_per_category(input_db)

    samples_num = {}
    for key in keys_per_cat.keys():
        if percent_in_first < 1:
            first = round(percent_in_first * len(keys_per_cat[key]))
        else:
            first = percent_in_first
        second = len(keys_per_cat[key])
        samples_num[key] = (0, int(first), second)

    if shuffle:
        for key in keys_per_cat.keys():
            random.shuffle(keys_per_cat[key])

    env_in = lmdb.open(input_db, readonly=True, map_size=map_size)
    with env_in.begin() as txn_in:
        cursor_in = txn_in.cursor()
        for db_num, output_db in enumerate(output_dbs):
            env_out = lmdb.open(output_db, readonly=False, map_size=map_size)
            idx = 0

            with env_out.begin(write=True) as txn_out:
                for cat, sample_keys in keys_per_cat.items():
                    for sample_idx in xrange(samples_num[cat][db_num], samples_num[cat][db_num + 1]):
                        key = keys_per_cat[cat][sample_idx]

                        txn_out.put(db.str_id(idx), cursor_in.get(key))
                        idx += 1
                        if idx % 1000 == 0:
                            print 'Processed {0:8} samples for db: {1}'.format(idx, output_db)

            env_out.close()
    env_in.close()

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print 'merge_dbs.py input_db output_db1 output_db2 percent_in_first'
        sys.exit(1)

    input_db = sys.argv[1]
    output_dbs = sys.argv[2:-1]
    percent_in_first = float(sys.argv[-1])
    split_dbs(input_db, output_dbs, percent_in_first)