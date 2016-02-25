import sys


def wait_bar(pre_msg, post_msg, counter, limit):
    print '{0} {1}/{2} {3}\r'.format(pre_msg, counter, limit, post_msg),
    sys.stdout.flush()
