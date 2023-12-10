"""
This script preprocesses the product data.

This script contatins following functions:
    *restructure_data - returns a list containing the products instead of a JSON
    *set_product_brand - ensures each product in the data set has a brand attribute
    *standardize_product_units - replaces the units in the title and the key-value pairs of the product to a standardized format
"""

import re

def restructure_data(products):
    """This method restructures the data as JSON file to a list.
    
    Parameters
    ----------
    products: Any
        list containing the data on the products
    
    Returns
    -------
    list
        list containing the data
    """
    
    # Initialize the list in which we store the data on the products
    restructured_data = []

    # Store data in the list
    for product in products:
        # Get the number of products with the same modelID
        num_identical_products = len(products[product])
        
        for i in range(num_identical_products):
            restructured_data.append(products[product][i])
            
    # Return the restructured data
    return(restructured_data)

def set_product_brand(products: list):
    """This method replaces the units in the title and the key-value pairs of the products.
    
    Parameters
    ----------
    products: list
        list containing the data on the products
    
    Returns
    -------
    list
        data on the products with the correct units
    """
    
    # Ensure that each product has an attribute 'Brand'
    for product in products:
        if 'Brand' in product['featuresMap']:
            product['featuresMap']['Brand'] = product['featuresMap']['Brand'].lower()
        elif 'Brand Name' in product['featuresMap']:
            product['featuresMap']['Brand'] = product['featuresMap'].pop('Brand Name').lower()
        else:
            # Get all words from the title
            title = product['title'].lower()
            title_words = re.findall(r'\b[A-Za-z]+\b', title)

            # Use the format of the title to find the brand
            product['featuresMap']['Brand'] = next(title_word for title_word in title_words if title_word not in ['newegg', 'com', 'refurbished', 'open', 'box'])
        
        # Deal with brands that have multiple names or spelling mistakes
        if product['featuresMap']['Brand'] == 'lg electronics':
            product['featuresMap']['Brand'] = 'lg'
        elif product['featuresMap']['Brand'] == 'jvc tv':
            product['featuresMap']['Brand'] = 'jvc'
        elif product['featuresMap']['Brand'] == 'pansonic':
            product['featuresMap']['Brand'] = 'panasonic'

    # Return products
    return(products)

def standardize_product_units(products: list):
    """This method replaces the units in the title and the key-value pairs of the product to a standardized format.
    
    Parameters
    ----------
    products: list
        list containing the data on the products
    
    Returns
    -------
    list
        data on the products with the correct units
    """
    def replace_unit(match):
        # Dictionary which stores the units that should be replaced
        units_to_replace = {'hz' : ['hz', '-hz', 'hertz'],
                            'inch' : ['inch', '-inch' 'inches', '-inches', '\"'],
                            'nit' : ['nit', 'cd/m\u00c2\u00b2', 'cd/m\u00b2', 'cd/m2'],
                            'watt': ['watt', 'watts', 'w'],
                            'lb': ['lb', 'lb.', 'lbs', 'lbs.'],
                            'hours': ['hours', 'hour', 'hrs']}
        
        # Retrieve the value and unit of the word 
        number = match.group(1)
        unit = match.group(2).lower()

        # Replace the 'incorrect' unit of the word with the correct unit
        for correct_unit, incorrect_units in units_to_replace.items():
            for incorrect_unit in incorrect_units:
                if incorrect_unit in unit:
                    return number + correct_unit
                
    # Replace the units in the product title and values from the key-value pairs
    for product in products:
        # Replace the units in the product title
        pattern = r'\b(\d+)\s?((?:-?)?[iI]nch(?:es?)?|\"|[Ww](?:atts?)?|[lL][bB](?:s?)\.?|(?:-?)?[Hh][Zz]|[Hh]ertz|[Nn]it|cd/m(?:\u00b2|\u00c2\u00b2|2?)|[Hh]our(?:s?)?|hrs)\b'
        product['title'] = re.sub(pattern , replace_unit, product['title'])    

        # Replace the units in the values from the key-value pairs
        product_features = product['featuresMap']
        for product_feature, product_value in product_features.items():
            product_features[product_feature] = re.sub(pattern , replace_unit, product_value)  
                  
    # Return the data
    return(products)