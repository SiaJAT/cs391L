#!/bin/bash


# in actuality, divided experiments into 10 separate processes running
python eigen_reduced_exps.py 0 1000
python eigen_reduced_exps.py 1000 2000
python eigen_reduced_exps.py 2000 3000
python eigen_reduced_exps.py 3000 4000
python eigen_reduced_exps.py 4000 5000
python eigen_reduced_exps.py 5000 6000
python eigen_reduced_exps.py 6000 7000
python eigen_reduced_exps.py 7000 8000
python eigen_reduced_exps.py 8000 9000
python eigen_reduced_exps.py 9000 10000

# output results 
python eval_classif_acc.py L1 > classification_L1.csv
python eval_classif_acc.py L2 > classification_L2.csv
