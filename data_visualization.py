import pandas as pd

def read_csv(src: str):
    return pd.read_csv(src, delimiter=";", engine="python")

collect_data = read_csv("./raw-datasets/colheita-de-sangue-total.csv")
donators_data = read_csv("./raw-datasets/dadores-de-sangue.csv")
storage_data = read_csv("./raw-datasets/reservas.csv")

collect_data.info()
donators_data.info()
storage_data.info()