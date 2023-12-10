"""
This script extracts model words from the titles and key-value pairs of the products.
Then, it constructs a characteristic matrix which can be applied to the minhash
algorithm.

This script contatins following functions:
    *get_model_words - returns all the model words extracted from the titles and key-value pairs
    *get_characteristic_matrix - returns the characteristic matrix used for the minhash algorithm
"""

import numpy as np
import re

def get_model_words(products: list):
    """This method retrieve the models words from the titles and the key-value pairs.
    
    Parameters
    ----------
    products: list
        list containing the data on the products
    
    Returns
    -------
    set
        set containing the model words
    """

    # Get the title and the features of the products
    all_product_titles = [product['title'] for product in products]
    all_product_values = [tuple(product['featuresMap'].values()) for product in products]
    
    # Create an empty set for the model words
    model_words = set()
    
    # Add model words from the titles to the universal set
    for product_title in all_product_titles:
        # Extract the model words from the title
        pattern = re.compile(r'\b[a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*\b')

        # Add model words to the universal set
        model_words.update(tuple([word.group() for word in pattern.finditer(product_title)]))

    # Add model words from the values to the universal set
    for product_values in all_product_values:
        for product_value in product_values:
            # Extract the model words from the value
            pattern = re.compile(r'^\d+(\.\d+)?[a-zA-Z]+$|^\d+(\.\d+)?$')
            model_words_product = [word.group() for word in pattern.finditer(product_value)]
            
            if len(model_words_product) > 0:
                # Remove the non-numerical part from the model word
                model_words_product = re.sub(r'[^0-9.]', '', model_words_product[0])

                # Add model words to the universal set containing the model words
                model_words.add(model_words_product)
    
    # Delete model words with less than two characters
    model_words = model_words.difference(set([model_word for model_word in model_words if len(model_word) < 2]))

    # Return the model words
    return(model_words)

def get_characteristic_matrix(products: list):
    """This method constructs the characteristic matrix based on the model words.
    
    Parameters
    ----------
    products: list
        list containing the data on the products
    
    Returns
    -------
    numpy.array
        the characteristic matrix
    """
    # Get all model words
    model_words = get_model_words(products)

    # Create a matrix containing zeroes
    num_rows = len(model_words)
    num_cols = len(products)
    characteristic_matrix = np.zeros((num_rows, num_cols))

    # Update the matrix
    for p, product in enumerate(products):
        # Check which model words are present in the title or in the values
        for w, model_word in enumerate(model_words):
            if model_word in product['title'] or model_word in list(product['featuresMap'].values()):
                characteristic_matrix[w][p] = 1
                    
    # Return the characteristic matrix
    return(characteristic_matrix)