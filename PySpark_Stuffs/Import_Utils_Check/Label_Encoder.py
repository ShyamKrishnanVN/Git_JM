from pyspark.ml.feature import StringIndexer

def label_encoder(df, input_col, output_col):
    indexer = StringIndexer(inputCol=input_col, outputCol=output_col);
    indexed_df = indexer.fit(df).transform(df);
    return indexed_df