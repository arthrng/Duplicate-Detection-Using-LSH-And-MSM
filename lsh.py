"""
This script performs locality-sensitive hashing to select candidate pairs for the
MSM algorithm.

This script contatins following functions:
    *get_num_bands - returns all the number of possible bands given the numbers of rows in the signature matrix
    *hash_columns_to_buckets - for each band, the columns are hashed to a bucket
    *get_candidate_pairs - returns all the candidate pairs
    *lsh - perform lsh to determine the candidate pairs
"""

import numpy as np
from itertools import combinations

def get_num_bands(num_rows: int):
    """This method computes all possible number of bands given the number of rows in the signature matrix.
    
    Parameters
    ----------
    num_rows: int
        number of rows in the signature matrix
    
    Returns
    -------
    list
        all possible number of bands
    """

    # Initialize the list which contains all possible number of bands
    num_bands = []

    for r in range(2, num_rows):
        # Compute the number of bands
        b = int(num_rows / r)

        # Check if the relation n = b * r holds
        if num_rows == b * r:
            num_bands.append(b)
    
    # Return the list which contains all possible number of bands
    return(num_bands)

def hash_columns_to_buckets(bands: list, observations: list):
    """This method hashes the columns to buckets for each band.
    
    Parameters
    ----------
    bands: list
        list containing all the bands
    observations: list
        list containing the position of the products in the original data set
    
    Returns
    -------
    list
        a list of buckets corresponding to each band
    """

    # Create an empty list which will contain the buckets for each band
    buckets_list = []

    for band in bands:
        # Create an empty dictionary which will store all the buckets
        buckets = dict()

        # Hash the columns of the band to a bucket
        for c in range(band.shape[1]):
            # Retrieve column
            column = band[:, c]

            # Hash the column to a bucket
            hash = tuple(column)

            # Store the column in a bucket
            if hash in buckets.keys():
                buckets[hash].append(observations[c])
            else:
                buckets[hash] = [observations[c]]

        # Append the buckets to the list of buckets
        buckets_list.append(buckets)

    # Return the dictionary of buckets
    return(buckets_list)

def get_candidate_pairs(buckets_list: list):
    """This method determines the candidate pairs based on the buckets.
    
    Parameters
    ----------
    buckets_list: list
        a list of buckets corresponding to each band
    
    Returns
    -------
    numpy.array
        the matrix containing the candidate pairs
    """

    # Create an empty matrix in which we store the candidate pairs
    candidate_pairs = set()

    # Identify the candidate pairs
    for buckets in buckets_list:
        for bucket in buckets.values():
            for pair in list(combinations(bucket, 2)):
                candidate_pairs.add(pair)

    # Return the candidate pairs
    return(candidate_pairs)
                    
def lsh(signature_matrix: np.array, observations: list, num_bands: int):
    """This method performs locality-sensitive hashing to find candidate pairs in the signature matrix.
    
    Parameters
    ----------
    signature_matrix: numpy.array
        the signature matrix
    observations: list
        list containing the position of the products in the original data set
    num_bands: int
        the number of bands we divide the matrix into
    
    Returns
    -------
    numpy.array
        the matrix containing the candidate pairs
    """

    # Divide the signature matrix into bands containing r rows
    bands = np.split(signature_matrix, num_bands, axis=0)

    # Create a list which will contain the dictionary of buckets for each band
    buckets_list = hash_columns_to_buckets(bands, observations)

    # Determine the candidate pairs
    candidate_pairs = get_candidate_pairs(buckets_list)

    # Return the candidate pairs
    return(candidate_pairs)