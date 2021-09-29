
# Data handling dependencies
import pandas as pd
import numpy as np

def load_product_titles(path_to_products):
    df = pd.read_csv(path_to_products)
    df = df.dropna()
    product_list = df['title'].to_list()
    return product_list
