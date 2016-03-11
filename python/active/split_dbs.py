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

    keys_per_cat = keys_per_category(input_db)

    samples_num = {}
    for key in keys_per_cat.keys():
        first = percent_in_first
        if first < 1:
            first = round(first * len(keys_per_cat[key]))

        second = len(keys_per_cat[key])
        samples_num[key] = (0, int(first), second)

        if shuffle:
                random.shuffle(keys_per_cat[key])

    for db_num, output_db in enumerate(output_dbs):
        sample_keys = []
        for cat, samples_per_cat in keys_per_cat.items():
            for sample_idx in xrange(samples_num[cat][db_num], samples_num[cat][db_num + 1]):
                sample_keys.append(samples_per_cat[sample_idx])

        db.extract_samples(input_db, output_db, sample_keys, shuffle=shuffle)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print 'merge_dbs.py input_db output_db1 output_db2 percent_in_first'
        sys.exit(1)

    input_db = sys.argv[1]
    output_dbs = sys.argv[2:-1]
    percent_in_first = float(sys.argv[-1])
    split_dbs(input_db, output_dbs, percent_in_first)