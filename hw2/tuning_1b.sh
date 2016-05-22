#!/bin/bash

nohup python ica.py 0.4 2600 5 &> trace_0p4bp_2600r5m.log &
nohup python ica.py 0.4 2700 5 &> trace_0p4bp_2700r5m.log &
nohup python ica.py 0.4 2800 5 &> trace_0p3bp_2800r5m.log &
nohup python ica.py 0.4 2900 5 &> trace_0p4bp_2900r5m.log &
nohup python ica.py 0.4 3000 5 &> trace_0p4bp_3000r5m.log &
