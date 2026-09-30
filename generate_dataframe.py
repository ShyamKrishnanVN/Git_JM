
import pandas as pd

def generate_dataframe():
    """
    Creates and returns a sample pandas DataFrame.
    """
    data = {
        "id": [1, 2, 3, 4],
        "name": ["Alice", "Bob", "Charlie", "David"],
        "age": [25, 30, 35, 40],
        "city": ["Bangalore", "Mumbai", "Delhi", "Hyderabad"]
    }

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    df = generate_dataframe()
    print(df)
