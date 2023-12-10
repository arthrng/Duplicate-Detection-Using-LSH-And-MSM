"""
This script performs the Multi-component Similarity Method to find the duplicate
products.

This script contatins following functions:
    *compute_qgram_similarity - returns the q-gram similarity between two string
    *extract_model_words - extracts the model word from a given string
    *compute_similarity_values - returns the similarity between the values with similar keys and the values with non-similar keys.
    *compute_average_levenshtein_similarity - returns the average Levenshtein similarity between two sets of words
    *compute_similarity_title - returns the similarity between two product titles
    *get_dissimilarity_matrix  - returns the matrix containing the pairwise dissimilarities between the products
    *get_pairs - returns the duplicate products found by the agglomerative clustering algorithm
"""

import numpy as np
import math
import re
import nltk
import sys
from sklearn.cluster import AgglomerativeClustering
from strsimpy.qgram import QGram
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
from itertools import combinations
from nltk.corpus import stopwords
nltk.download('stopwords')

def compute_qgram_similarity(val1: str, val2: str, q: int):
    """This method computes the q-gram similarity between two strings.
    
    Parameters
    ----------
    val1: str
        first string
    val2: str
        second string
    q: int
        the size of the q-grams
    
    Returns
    -------
    int
        the q gram similarity between two strings
    """

    # Initialize q-gram object
    qgram = QGram(q)

    # Compute the q-gram similarity
    return((len(val1) + len(val2) - qgram.distance(val1, val2)) / (len(val1) + len(val2)))

def extract_model_words(string: str, model_words: set):
    """This method extracts the model words from a certain string.
    
    Parameters
    ----------
    string: str
        a string
    model_words: set
        set containing the model words
    
    Returns
    -------
    set:
        the model words that are present in the given string
    """

    # Create an empty set in which we store the model words present in the string
    extracted_model_words = set()

    # Extract the model word from the string
    for model_word in model_words:
        if model_word in string:
            extracted_model_words.add(model_word)
    
    # Return the model words in the string
    return(extracted_model_words)

def compute_similarity_values(product_i: dict, product_j: dict,  model_words: set, q: int, gamma: float):
    """This method computes the similarity between the values with similar
    keys and the values with non-similar keys.
    
    Parameters
    ----------
    product_i: dict
        dictionary containing information about product i
    product_j: dict
        dictionary containing information about product j
    model_words: set
        set containing the model words
    q: int
        the size of the q grams
    gamma: float
        minimum similarity for two keys to be matching 
    
    Returns
    -------
    float
        similarity between the values with similar keys and the values with non-similar keys,
        and the number of times we got a matching key
    """

    # Initialize variables for computing the similarity of the matching keys
    mk_similarity = 0
    similarity = 0
    num_matches = 0
    weight_matches = 0

    # Retrieve the keys-value pairs of the two products
    kvp_i = product_i['featuresMap']
    kvp_j = product_j['featuresMap']

    # Initialize the sets storing the non-matching keys
    nmk_i = list(kvp_i.keys())
    nmk_j = list(kvp_j.keys())

    # Compare the values of the key-value pairs that have similar keys
    for key_i in kvp_i.keys():
        for key_j in kvp_j.keys():
            # Compute the q-gram similarity between the two keys
            key_sim = compute_qgram_similarity(key_i.lower(), key_j.lower(), q) 

            if key_sim > gamma:
                # Compute the q-gram similarity between the two values
                value_sim = compute_qgram_similarity(kvp_i[key_i].lower(), kvp_j[key_j].lower(), q)

                # Update similarity
                similarity += key_sim * value_sim

                # Update the number of matches and the weight of matches
                num_matches += 1
                weight_matches += key_sim
                
                # Remove the keys from nmk1 and nmk2
                if key_i in nmk_i:
                    nmk_i.remove(key_i)
                if key_j in nmk_j:
                    nmk_j.remove(key_j)

    # Compute the average similarity
    if weight_matches > 0:
        mk_similarity = similarity / weight_matches
    
    # Extract the model words from the values of the non-matching keys
    model_words_i = set()
    for key in nmk_i:
        model_words_i.update(extract_model_words(kvp_i[key], model_words))

    model_words_j = set()
    for key in nmk_j:
        model_words_j.update(extract_model_words(kvp_j[key], model_words))

    # Compute the percentage of overlapping model words
    if model_words_i or model_words_j:
        nmk_similarity = len(model_words_i.intersection(model_words_j)) / len(model_words_i.union(model_words_j))
    else:
        nmk_similarity = 0

    # Return mk_similarity and nmk_similarity
    return(mk_similarity, nmk_similarity, num_matches) 

def compute_average_levenshtein_similarity(X: set, Y: set, normalized_levenshtein: NormalizedLevenshtein()):
    """This method average Levenshtein similarity between two sets of words.
    
    Parameters
    ----------
    X: set
        first set containing words
    Y: set
        second set containing words
    normalized_levenshtein: NormalizedLevenshtein
        Normalized Levenshtein object
    
    Returns
    -------
    int
        the average Levenshtein similarity between two sets of words
    """

    # Compute the numerator and denominator in the average Levenshntein similarity
    numerator = 0
    denominator = 0
    for x in X:
        for y in Y:
            numerator += (1 - normalized_levenshtein.distance(x, y)) * (len(x) + len(y))
            denominator += len(x) + len(y)

    # Return the average Levenshntein similarity    
    return(numerator / denominator)

def compute_similarity_title(title_i: str, title_j: str, model_words: set, alpha: float, beta: float, delta: float, eps: float):
    """This method uses the Title Model Word Method to compute the similarity
    between two product titles.
    
    Parameters
    ----------
    title_i: str
        title of product i
    title_j: str
        title of product j
    model_words: set
        set containing the model words
    alpha: float
        minimum cosine similarity for two products to be labeled as equal
    beta: float
        weight placed on the cosine similarity between the two titles
    delta: float 
        weight placed on the average Levenshtein similarity
    eps: float
        maximum threshold of the distance for two model words to be considered similar

    Returns
    -------
    float
        similarity between the title of product i and product j
    """
    
    # STEP 1: Compute the cosine similarity between both titles
    # Remove special characters from the titles
    filtered_title_i = re.sub(r'[^a-zA-Z0-9. ]', ' ', title_i)
    filtered_title_j = re.sub(r'[^a-zA-Z0-9. ]', ' ', title_j)

    # Remove stopwords and other words that could cause noise
    words_to_remove = set(stopwords.words('english')).union(set(['best', 'buy', 'newegg.com', 'thenerdsnet', 'refurbished', 'open', 'box']))
    filtered_title_i = [word for word in filtered_title_i.split(' ') if word.lower() not in words_to_remove]
    filtered_title_j = [word for word in filtered_title_j.split(' ') if word.lower() not in words_to_remove]

    # Lower all the letters
    filtered_title_i_lowered = set(map(str.lower, filtered_title_i))
    filtered_title_j_lowered = set(map(str.lower, filtered_title_j))

    # Calculate the cosine similarity between the two titles
    title_cosine_similarity = len(filtered_title_i_lowered.intersection(filtered_title_j_lowered)) / (math.sqrt(len(filtered_title_i_lowered)) * math.sqrt(len(filtered_title_j_lowered)))

    # Return a similarity of 1 if the cosine similarity is above a certain threshold
    if title_cosine_similarity > alpha:
        return(1)
    
    # STEP 2: Compute a weighted similarity between the two titles
    # Extract model words from the title
    model_words_i = extract_model_words(filtered_title_i, model_words)
    model_words_j = extract_model_words(filtered_title_j, model_words)

    # Check if the two products can be identified as different
    for model_word_i in model_words_i:
        for model_word_j in model_words_j:
            # Extract the numerical and non-numerical parts from both model words
            model_word_i_numerical = ''.join(re.findall(r'(\d+)', model_word_i))
            model_word_j_numerical = ''.join(re.findall(r'(\d+)', model_word_j))
            model_word_i_nonnumerical = ''.join(re.findall(r'(\D+)', model_word_i))
            model_word_j_nonnumerical = ''.join(re.findall(r'(\D+)', model_word_j))

            # Check if we find that non-numeric characters are approximately the same and numeric characters are not the same
            if NormalizedLevenshtein().distance(model_word_i_nonnumerical, model_word_j_nonnumerical) < eps and model_word_i_numerical != model_word_j_numerical:
                return(-1)
            
    # Compute initial similarity between the two titles
    title_similarity = beta * title_cosine_similarity + (1 - beta) * compute_average_levenshtein_similarity(list(filtered_title_i), list(filtered_title_j), NormalizedLevenshtein())

    # STEP 3: Update the initial similarity
    # Initialize variables
    has_match = False
    numerator = 0
    denominator = 0

    for model_word_i in model_words_i:
        for model_word_j in model_words_j:
            # Extract the numerical and non-numerical parts from both model words
            model_word_i_numerical = ''.join(re.findall(r'(\d+)', model_word_i))
            model_word_j_numerical = ''.join(re.findall(r'(\d+)', model_word_j))
            model_word_i_nonnumerical = ''.join(re.findall(r'(\D+)', model_word_i))
            model_word_j_nonnumerical = ''.join(re.findall(r'(\D+)', model_word_j))

            # Check if we find that non-numeric characters are approximately the same and numeric characters are the same
            if NormalizedLevenshtein().distance(model_word_i_nonnumerical, model_word_j_nonnumerical) < eps and model_word_i_numerical == model_word_j_numerical:
                has_match = True
                
                # Update Levenshtein similarity
                numerator += (1 - NormalizedLevenshtein().distance(model_word_i, model_word_j)) * (len(model_word_i) + len(model_word_j))
                denominator += len(model_word_i) + len(model_word_j)

    # Check if there at least one match between the model words            
    if has_match:
        title_similarity = delta * (numerator/denominator) + (1 - delta) * title_similarity

    # Return the similarity between the two titles
    return(title_similarity)

def get_dissimilarity_matrix(products: list, candidate_pairs: set, model_words: set, q: int, alpha: float, beta: float, gamma: float, delta: float, eps: float, mu: float):
    """This method constructs the matrix containing the pairwise dissimilarities between all the products.
    
    Parameters
    ----------
    products: list
        list containing the data on the products
    candidate_pairs: set
        set containing the candidate pairs
    model_words: set
        set containing the model words
    alpha: float
        minimum cosine similarity for two products to be labeled as equal
    beta: float
        weight placed on the cosine similarity between the two titles
    delta: float 
        weight placed on the average Levenshtein similarity
    gamma: float
        minimum similarity for two keys to be matching 
    eps: float
        maximum threshold of the distance for two model words to be considered similar
    mu: float
        weight placed on the similarity between the two product titles
    
    Returns
    -------
    numpy.array
        the dissimilarity matrix
    """

    # Create the dissimilarity matrix
    dissimilarity_matrix = np.ones((len(products), len(products))) * sys.maxsize

    # Update the dissimilarity matrix
    for candidate_pair in candidate_pairs:
        # Get products from the pair
        i = candidate_pair[0]
        j = candidate_pair[1]

        # Check if the web shop is different or the brand is the same
        if products[i]['shop'] != products[j]['shop'] and products[i]['featuresMap']['Brand'] == products[j]['featuresMap']['Brand']:           
            # Compute the similarity of the key-value pairs
            mk_similarity, nmk_similarity, num_matches = compute_similarity_values(products[i], products[j], model_words, q, gamma)

            # Compute the TMWM similarity
            tmwm_similarity = compute_similarity_title(products[i]['title'], products[j]['title'], model_words, alpha, beta, delta, eps)

            # Calculate the final similarity
            if tmwm_similarity == -1:
                theta_1 = num_matches / min(len(products[i]['featuresMap']), len(products[j]['featuresMap']))
                theta_2 = 1 - theta_1
                final_similarity = theta_1 * mk_similarity + theta_2 * nmk_similarity
            else:
                theta_1 = (1 - mu) * (num_matches / min(len(products[i]['featuresMap']), len(products[j]['featuresMap']))) 
                theta_2 = 1 - mu - theta_1
                final_similarity = theta_1 * mk_similarity + theta_2 * nmk_similarity + mu * tmwm_similarity

            # Update the dissimilarity matrix
            dissimilarity_matrix[i][j] = 1 - final_similarity
            dissimilarity_matrix[j][i] = dissimilarity_matrix[i][j]
    
    # Return the dissimilarity matrix
    return(dissimilarity_matrix)

def get_pairs(dissimilarity_matrix: np.array, distance_threshold: float):
    """This method determines the duplicate products based on agglomerative clustering.
    
    Parameters
    ----------
    dissimilarity_matrix: numpy.array
        the matrix containing the pairwise similarities between all the products
    distance_threshold: float
        the linkage distance threshold above which two clusters will not be merged
    
    Returns
    -------
    set:
        the duplicate pairs
    """

    # Perform clustering
    cluster_model = AgglomerativeClustering(affinity='precomputed', linkage='average', distance_threshold=distance_threshold, n_clusters=None)
    cluster_model.fit(dissimilarity_matrix)

    # Store the clusters in a dictionary
    clusters = dict()
    for product, cluster in enumerate(cluster_model.labels_):
        if cluster in clusters.keys():
            clusters[cluster].append(product)
        else:
            clusters[cluster] = [product]

    # Construct the pairs
    pairs = set()
    for products in clusters.values():
        for pair in list(combinations(products, 2)):
            pairs.add(pair)

    # Return the pairs
    return(pairs)