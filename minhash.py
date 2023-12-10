"""
This script computes the signature of each product using the minhashing 
technique.

This script contatins following functions:
    *generate_hash_function - returns a given number of hash functions used for minhashing
    *minhash - returns the signature matrix produced by the minhash algorithm
"""

import numpy as np
import sympy as sp
import random

def generate_hash_function(num_hash_functions: int):
    """This method generates the hash functions for the minhash algorithm.
    
    Parameters
    ----------
    num_hash_functions: int
        number of hash functions to be generated
    
    Returns
    -------
    list
        list of hash functions
    """

    # Create a list to store the hash functions
    hash_functions = []

    # Generate p by selecting a prime number p > num_hash_functions
    p = num_hash_functions + 1
    while not sp.isprime(p):
        p += 1

    for n in range(num_hash_functions):
        # Generate two random integers a and b
        a = random.randint(1, p - 1)
        b = random.randint(0, p - 1)

        # Define the hash function based on the parameter a, b and p
        def hash_function(x, a=a, b=b, p=p):
            return ((a * x + b) % p)      
        
        # Store the hash function in a list
        hash_functions.append(hash_function)
    
    # Return the list with hash functions
    return(hash_functions)

def minhash(characteristic_matrix: np.array, num_hash_functions: int):
    """This method generates a signature matrix for a given characteristic matrix.
    
    Parameters
    ----------
    characteristic_matrix: numpy.array
        the characteristic matrix
    num_hash_functions: int
        number of hash functions to be generated
    
    Returns
    -------
    numpy.array
        the signature matrix
    """

    # Create the signature matrix
    num_rows = num_hash_functions
    num_cols = characteristic_matrix.shape[1]
    signature_matrix = np.ones((num_rows, num_cols)) * np.inf
    
    # Generate the hash functions
    hash_functions = generate_hash_function(num_hash_functions)

    # Update the signature matrix
    for r in range(characteristic_matrix.shape[0]):
        for c in range(characteristic_matrix.shape[1]):
            if characteristic_matrix[r][c] == 1:
                for n in range(num_hash_functions):
                    # Compute hash value using the hash function
                    hash = hash_functions[n](r)

                    # Update entry in the signature matrix if the hash given by the hash function is smaller 
                    signature_matrix[n][c] = min(hash, signature_matrix[n][c])

    # Return signature matrix
    return(signature_matrix)