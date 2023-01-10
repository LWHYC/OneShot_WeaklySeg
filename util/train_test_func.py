# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import tensorflow as tf
from data_process.data_process_func import *
import sys

def process_bar(num, total):
    rate = float(num)/total
    ratenum = int(100*rate)
    r = '\r[{}{}]{}%'.format('*'*ratenum,' '*(100-ratenum), ratenum)
    sys.stdout.write(r)
    sys.stdout.flush()