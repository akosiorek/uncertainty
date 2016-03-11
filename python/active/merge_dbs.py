#!/usr/bin/env python
import sys
import random

import lmdb

import db


def merge_dbs(input_dbs, output_db, shuffle=True):

    map_size = 2**40

    total_size = sum([db.size(input_db) for input_db in input_dbs])
    ids = range(total_size)
    if shuffle:
        random.shuffle(ids)

    env_out = lmdb.open(output_db, readonly=False, map_size=map_size)
    idx = 0

    for input_db in input_dbs:
        env_in = lmdb.open(input_db, readonly=True, map_size=map_size)

        with env_out.begin(write=True) as txn_out:
            with env_in.begin() as txn_in:
                for key, data in txn_in.cursor():
                    txn_out.put(db.str_id(ids[idx]), data)
                    idx += 1
                    if idx % db.LOG_EVERY == 0:
                        print 'Processed {0} samples'.format(idx)

        env_in.close()
    env_out.close()

    if idx % db.LOG_EVERY != 0:
        print 'Processed {0} samples'.format(idx)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print 'merge_dbs.py input_db1 input_db2 .... input_dbN output_db'
        sys.exit(1)

    input_dbs = sys.argv[1:-1]
    output_db = sys.argv[-1]
    merge_dbs(input_dbs, output_db)