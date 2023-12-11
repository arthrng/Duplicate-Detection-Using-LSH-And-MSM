"""
This script evaluates LSH and MSM.

This script contatins following functions:
    *get_duplicate_pairs - returns all the correct duplicate products
    *resample - splits the data into a training and set set
    *compute_recall_and_precision - returns recall and precision of the duplicates found by MSM
    *compute_f1 - returns the F1 score of the duplicates found by MSM
    *compute_pair_quality_and_completeness - returns the pair completeness and pair quality of the candidate pairs given by LSH
    *compute_f1star - returns the F1* score of the candidate pairs given by LSH
    *plot_metric - plots all the metrics on a graph
    *evaluate - evaluates LSH and MSM on a given number of bootstraps
"""

import json
import numpy as np
import random
from itertools import combinations
from matplotlib import pyplot as plt
import preprocessing
import characteristic
import minhash
import lsh
import msm

def get_duplicate_pairs(products: list, possible_pairs: set):
    """This method gets the duplicate pairs.
    
    Parameters
    ----------
    products: list
        list containing the data on the products
    possible_pairs: set
        set containing all possible pairs
    
    Returns
    -------
    set
        set containing all the duplicate pairs
    """

    # Initialize the set containing the correct pairs
    duplicate_pairs = set()

    # Get all the pairs with a matching modelID
    for possible_pair in possible_pairs:
        # Get the products
        i = possible_pair[0]
        j = possible_pair[1]

        # Add the pair to the set of correct pairs if they have matching modelID
        if products[i]['modelID'] == products[j]['modelID']:
            duplicate_pairs.add((i, j))
    
    # Return the correct pairs
    return(duplicate_pairs)

def resample(products):
    """This method splits the data into a training and test set.
    
    Parameters
    ----------
    products: list
        list containing the data on the products
    
    Returns
    -------
    list
        two lists containing the indices for the training samples and test samples
    """

    # Assign observations to the training and test set
    indices = list(range(0, len(products)))
    train_observations = random.choices(indices, k=len(products))
    test_observations = list(set(indices).difference(set(train_observations)))

    # Store the training and test products in a list
    train_products = [products[i] for i in train_observations]
    test_products = [products[i] for i in test_observations]

    # Return the training and test set
    return(train_observations, train_products, test_observations, test_products)

def compute_recall_and_precision(duplicate_pairs: set, predicted_pairs: set):
    """This method computes the recall and precision.
    
    Parameters
    ----------
    duplicate_pairs: set
        set containing the correct duplicate pairs
    predicted_pairs: set
        set containing the duplicate pairs found by MSM
    
    Returns
    -------
    int
        recall and precision
    """

    # Compute TP, FP and FN
    tp = len(duplicate_pairs.intersection(predicted_pairs))
    fp = len(predicted_pairs) - tp
    fn = len(duplicate_pairs) - tp

    # Compute recall
    if tp > 0 or fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0

    # Compute precision
    if tp > 0 or fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    # Return the recall and precision
    return(recall, precision)

def compute_f1(recall: float, precision: float):
    """This method computes the F1-score.
    
    Parameters
    ----------
    recall: float
        recall
    precision: float
        precision
    
    Returns
    -------
    int
        the F1-score
    """

    # Compute F1-score
    if precision > 0 or recall > 0:
        return((2 * recall * precision) / (recall + precision))
    else:
        return(0)

def compute_pair_quality_and_completeness(duplicate_pairs: set, candidate_pairs: set):
    """This method computes the pair quality and pair completeness.
    
    Parameters
    ----------
    duplicate_pairs: set
        set containing the correct duplicate pairs
    candidate_pairs: set
        set containing the candidate pairs found by LSH
    
    Returns
    -------
    float
        the pair quality and pair completeness
    """

    # Compute the pair quality
    if len(candidate_pairs) > 0:
        pair_quality = len(duplicate_pairs.intersection(candidate_pairs)) / len(candidate_pairs)
    else:
        pair_quality = 0

    # Compute the pair completeness
    if len(duplicate_pairs) > 0:
        pair_completeness = len(duplicate_pairs.intersection(candidate_pairs)) / len(duplicate_pairs)
    else:
        pair_completeness = 0

    # Return the pair quality and pair completeness
    return(pair_quality, pair_completeness)

def compute_f1star(pair_quality: float, pair_completeness: float):
    """This method computes the F1*-score.
    
    Parameters
    ----------
    pair_quality: int
        the pair quality
    pair_completeness: int
        the pair completeness  
    
    Returns
    -------
    int
        the F1*-score
    """

    if pair_quality > 0 or pair_completeness > 0:
        return((2 * pair_quality * pair_completeness) / (pair_quality + pair_completeness))
    else:
        return(0)

def plot_metric(x: list, y: list, x_label: str, y_label: str, file_name: str):
    """This method plots a metric on a graph.
    
    Parameters
    ----------
    x: list
        data on the x-axis
    y: list
        data on the y-axis
    x_label: str
        label of the x-axis
    y_label: str
        label of the y_axis
    file_name: str
        name of the picture file
    """

    # Show the grid
    plt.grid(True)

    # Show the plot
    plt.plot(x, y, color='black', linewidth=2)

    # Change the labels
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=18)

    # Change xlim of the plotW
    plt.xlim(left=-0.004)

    # Change tick sizes
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Save the figure
    plt.savefig('figures/' + file_name, bbox_inches='tight')

    # Show the plot
    plt.show()
    
def evaluate(num_bootstraps: int):
    """This method evaluates MSM on a test set.
    
    Parameters
    ----------
    pair_quality: int
        the number of bootstraps on which we evaluate MSM
    """

    # Import data
    file = open('data/TVs-all-merged.json')
    products = json.load(file)
    file.close()

    # Preprocess the data
    products = preprocessing.restructure_data(products)
    products = preprocessing.set_product_brand(products)
    products = preprocessing.standardize_product_units(products)

    # Get all possible number of bands
    bands = lsh.get_num_bands(1000)
    bands.reverse()

    # Intialize the distance thresholds
    distance_thresholds = np.arange(0, 1, 0.1)

    # Initialize the metrics
    metrics = np.zeros((len(bands), 7))

    # Run the program
    for n in range(num_bootstraps):
        print(f"------------------------------------\n BOOTSTRAP: {n + 1} \n------------------------------------")
         # Split the data into a train and test set
        train_observations, train_products, test_observations, test_products = resample(products)

        # TRAINING (Tuning the threshold of the clustering model)
        # Get all possible pairs in the training set
        train_all_pairs = set(combinations(set(train_observations), 2))

        # Get all the duplicate pairs in the training set
        train_duplicate_pairs = get_duplicate_pairs(products, train_all_pairs)

        # Get all the model words from the training set
        train_model_words = characteristic.get_model_words(train_products)

        # Create the characteristic matrix
        train_characteristic_matrix = characteristic.get_characteristic_matrix(train_products)

        # Create the signature matrix
        train_signature_matrix = minhash.minhash(train_characteristic_matrix, 1000)

        # Loop over possible number of bands
        for b, band in enumerate(bands):
            # Show the bands as debug message
            print('Bands: ' + str(band))

            # Perform LSH
            train_candidate_pairs = lsh.lsh(train_signature_matrix, train_observations, band)

            # Perform MSM
            train_dissimilarity_matrix = msm.get_dissimilarity_matrix(products, train_candidate_pairs, train_model_words, 3, 0.602, 0, 0.756, 0.4, 0.3, 0.65)

            # Find the best distance threshold for the clustering algorithm
            max_f1 = -np.inf
            for distance_threshold in distance_thresholds:             
                # Get the pairs
                train_predicted_pairs = msm.get_pairs(train_dissimilarity_matrix, distance_threshold)

                # Compute the F1 score
                train_recall, train_precision = compute_recall_and_precision(train_duplicate_pairs, train_predicted_pairs)
                train_f1 = compute_f1(train_recall, train_precision)

                # Update the highest F1 score
                if train_f1 > max_f1:
                    max_f1 = train_f1
                    best_threshold = distance_threshold

            # TESTING
            # Get all possible pairs in the test set
            test_all_pairs = set(combinations(set(test_observations), 2))

            # Get all the duplicate pairs in the test set
            test_duplicate_pairs = get_duplicate_pairs(products, test_all_pairs)

            # Get all the model words from the test set
            test_model_words = characteristic.get_model_words(test_products)

            # Create the characteristic matrix
            test_characteristic_matrix = characteristic.get_characteristic_matrix(test_products)

            # Create the signature matrix
            test_signature_matrix = minhash.minhash(test_characteristic_matrix, 1000)  

            # Perform LSH
            test_candidate_pairs = lsh.lsh(test_signature_matrix, test_observations, band)    

            # Perform MSM
            test_dissimilarity_matrix = msm.get_dissimilarity_matrix(products, test_candidate_pairs, test_model_words, 3, 0.602, 0, 0.756, 0.4, 0.3, 0.65)

             # Get the pairs
            test_predicted_pairs = msm.get_pairs(test_dissimilarity_matrix, best_threshold)

            # Evaluate the candidate pairs found by LSH
            test_pair_quality,  test_pair_completeness = compute_pair_quality_and_completeness(test_duplicate_pairs, test_candidate_pairs)
            test_f1star = compute_f1star(test_pair_quality, test_pair_completeness)
            
            # Evaluate the pairs found by the clustering algorithm
            frac_comparisons = len(test_candidate_pairs) / len(test_all_pairs)
            test_recall, test_precision = compute_recall_and_precision(test_duplicate_pairs, test_predicted_pairs)
            test_f1 = compute_f1(test_recall, test_precision)

            # Store the metrics in the array
            metrics_bands = np.array([frac_comparisons, test_recall, test_precision, test_f1, test_pair_quality, test_pair_completeness, test_f1star])
            metrics[b, :] += metrics_bands

    # Take the average of the metrics over all the bootstraps
    metrics = metrics / num_bootstraps

    # Plot all metrics
    plot_metric(metrics[:, 0], metrics[:, 1], 'Fraction of comparisons', 'Recall', 'recall.png')
    plot_metric(metrics[:, 0], metrics[:, 2], 'Fraction of comparisons', 'Precision', 'precision.png')
    plot_metric(metrics[:, 0], metrics[:, 3], 'Fraction of comparisons', 'F1 score', 'f1.png')
    plot_metric(metrics[:, 0], metrics[:, 4], 'Fraction of comparisons', 'Pair quality', 'pq.png')
    plot_metric(metrics[:, 0], metrics[:, 5], 'Fraction of comparisons', 'Pair completeness', 'pc.png')
    plot_metric(metrics[:, 0], metrics[:, 6], 'Fraction of comparisons', 'F1* score', 'f1star.png')

if __name__ == '__main__':
    evaluate(10)