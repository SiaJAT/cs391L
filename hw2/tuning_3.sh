#!/bin/bash

nohup python ica.py 0.2 20000 5 &> trace_0p2bp_20000r5m.log &
nohup python ica.py 0.2 50000 5 &> trace_0p2bp_50000r5m.log &
nohup python ica.py 0.2 100000 5 &> trace_0p2bp_100000r5m.log &