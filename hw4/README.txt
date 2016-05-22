Contents (this directory)
1. Results (Directory): contains result files
2. pca_svm.py, the main driving script, (Usage: python pca_svm.py) to run the experiments
3. Plotting (Directory)
4. time_sv

Contents (Results)
1. full_svm.txt, the results of the linear kernel SVM trained on 60000 examples, C=1.0
2. linear_svm.txt, the results of varying the C parameter for the linear SVM
3. rbf_svm.txt, the result of varying gamma for the RBF kernel SVM
4. size_svm.txt, the result of varying the size of the training set for the RBF kernel SVM
5. pca_svm.txt, the result of varying the number of principal compoonents (PCA-SVM) 
6. time_ovo.log, timiing experiment for training one vs. one SVM
7. time_ovr.log, timing experiment for training one vs. rest SVM

Contents (Plotting)
1. plotting.sh, (Usage: sh plotting.sh), run in directory containing plot_pca.gplot pca.data
2. pca.data, (data points from the PCA experiment)
3. plot_pca.gplot, simple GNUplot for plotting PCA data
5. pca_svm.pdf, a plot of the results from the PCA-SVM experiment

Recreating my experiments:
1. python pca_svm.py, to recreate experiments 1-5
2. nohup python time_svm.py ovo &> time_ovo.log, to recreate one vs. one timing experiment (may vary on different machine)
3. nohup python time_svm.py ovr &> time_ovr.log, to recreate one vs. rest timing experiment (may vary on different machine)
