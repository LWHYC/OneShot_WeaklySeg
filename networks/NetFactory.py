#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os
#os._exit(00)
import sys
sys.path.append(os.path.abspath(__file__)) 
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 
from PRNet import PRNet
from Unet import Unet
class NetFactory(object):

    @staticmethod
    def create(name):

        if name == 'PRNet':
            return PRNet

        if name == 'Unet':
            return Unet

        # add your own networks here
        print('unsupported network:', name)
        exit()
