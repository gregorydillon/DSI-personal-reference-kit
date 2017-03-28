import numpy as np
import pandas as pd

from zipfile import ZipFile
import cleanup as cln


def load_clean_from_zip(path):
    #TODO: unzip more safely?
    zf = ZipFile(path)
    df = pd.read_csv(path.replace('.zip', '.csv'))
    cleaned = cln.create_and_norm_categorical(df)

    return cleaned
