import pandas as pd
import os




for dirpath, dirnames, files in os.walk('channeltranscripts'):
    for name in files:
        if name.lower().endswith('.csv'):
            pd.read_csv(os.path.join(dirpath, name))




pd.read_csv()