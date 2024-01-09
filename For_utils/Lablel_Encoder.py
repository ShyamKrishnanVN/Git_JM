from sklearn.preprocessing import LabelEncoder

import pandas as pd

def label_encoder(df, input_col, output_col):
    le = LabelEncoder()
    df[output_col] = le.fit_transform(df[input_col])
    return df
