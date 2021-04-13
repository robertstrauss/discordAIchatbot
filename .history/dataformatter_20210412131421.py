import pandas as pd
import os

files = []

for dirpath, dirnames, files in os.walk('channeltranscripts'):
    for name in files:
        if name.lower().endswith('.csv'):
            print('reading', name)
            data = pd.read_csv(os.path.join(dirpath, name))




pd.read_csv()