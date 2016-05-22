#!/bin/bash

nohup python ica.py 0.5 2100 5 &> trace_0p5bp_2100r5m.log &
nohup python ica.py 0.5 2200 5 &> trace_0p5bp_2200r5m.log &
nohup python ica.py 0.5 2300 5 &> trace_0p5bp_2300r5m.log &
nohup python ica.py 0.5 2400 5 &> trace_0p5bp_2400r5m.log &
nohup python ica.py 0.5 2500 5 &> trace_0p5bp_2500r5m.log &
