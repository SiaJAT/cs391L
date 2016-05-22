#!/bin/bash

nohup python ica.py 0.5 2600 5 &> trace_0p5bp_2600r5m.log &
nohup python ica.py 0.5 2700 5 &> trace_0p5bp_2700r5m.log &
nohup python ica.py 0.5 2800 5 &> trace_0p5bp_2800r5m.log &
nohup python ica.py 0.5 2900 5 &> trace_0p5bp_2900r5m.log &
nohup python ica.py 0.5 3000 5 &> trace_0p5bp_3000r5m.log &
