# ---- human input ----
cd = 'YOUR_PRODUCT_ID'
code_choice_now = cd.upper()
# ----------------------------


"""
Simple Recommendation Engine
"""

# libraries
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from ast import literal_eval


# importing product dataset in CSV
products = pd.read_csv('../input/datasets-giolitti/products.csv', sep=";")
"""
USEFUL INFOS:
metadata -> products in our dataset
overview -> description in our dataset
title -> code in our dataset
"""

# convert to vector
tfidf = TfidfVectorizer()

# replace NaN with an empty string
products['description'] = products['description'].fillna('')

# construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(products['description'])

# save matrix infos
shape_m = tfidf_matrix.shape
feat_nam = tfidf.get_feature_names()

# compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# construct a reverse map of indices and movie titles
indices = pd.Series(products.index, index=products['code']).drop_duplicates()

# function that takes in product title as input and outputs most similar products
def get_recommendations(code, cosine_sim=cosine_sim):
    # get the index of the product that matches the title
    idx = indices[code]

    # get the pairwsie similarity scores of all products with that product
    sim_scores = list(enumerate(cosine_sim[idx]))

    # sort the products based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # get the scores of the 10 most similar products
    sim_scores = sim_scores[1:6]

    # get the products indices
    products_indices = [i[0] for i in sim_scores]

    # return the top 10 most similar products
    return products['code'].iloc[products_indices]

# save the recommendation based on product id
simple_recomm_eng = get_recommendations(code_choice_now))







"""
Content Based Reccomendation Engine
"""

# load keywords and subcategories
subcategory = pd.read_csv('../input/datasets-giolitti/subcategory.csv', sep=";")
types = pd.read_csv('../input/datasets-giolitti/type.csv', sep=";")

# remove rows with bad id IF NECESSARY
#metadata = metadata.drop([19730, 29503, 35587])

# convert id to int. which is required for merging
subcategory['code'] = subcategory['code'].astype('object')
types['code'] = types['code'].astype('object')
products['code'] = products['code'].astype('object')

# merge keywords and subcategories into your main metadata dataframe
products = products.merge(subcategory, on='code')
products = products.merge(types, on='code')


# parse the stringified features into their corresponding python objects
features = ['description', 'category', 'subcategory', 'type']

# function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        # check if value exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
       
for feature in features:
    products[feature] = products[feature].apply(clean_data)

products_only = products['description'] + ',' + products['category'] + ',' + products['subcategory'] + ',' + products['type']

# import CountVectorizer and create the count matrix
count = CountVectorizer()
count_matrix = count.fit_transform(products_only)

# compute the Cosine Similarity matrix based on the count_matrix
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# reset index of your main DataFrame and construct reverse mapping as before
products = products.reset_index()
indices = pd.Series(products.index, index=products['code'])



content_recomm_eng = get_recommendations(code_choice_now, cosine_sim2)
