# Duplicate Detection Using LSH And MSM #

This repository includes the code for the Multi-component Similarity Method (MSM), a method which can be utilized to detect duplicate products in a data set. To improve the efficiency of the MSM, we employ
Locality-Sensitive Hashing (LSH) as a preselection step to select candidate pairs.

## Scripts ##
The following scripts are included in the repository:
* characteristic.py - This script extract the model words from the product titles and key-value pairs. Then, it creates a binary representation of each product and stores them in a matrix.
* evaluation.py - This script evaluates the performance of MSM and LSH by bootstrapping the data, computing several metrics and drawing plots.
* lsh.py - This script performs LSH to find candidate pairs.
* minhash.py - This script creates a signature for each product using their binary representation, and stores the signatures in a matrix.
* msm.py - This scripts perform the MSM.
* preprocessing.py - This script preprocesses the data.

## How To Use ##
In order to evaluate MSM and LSH, one can simply run the evaluation.py script. It is important to make sure that you to change the path of the data set to the place where you have stored the file.
Several parameters can be modified e.g., the number of bootstraps and the number of columns in the signature matrix. When the script is finished
running
