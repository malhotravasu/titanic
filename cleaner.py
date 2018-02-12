''' Data cleaning and Fine Tuning '''
import re
import numpy as np

RE = re.compile(r", [a-zA-Z]*\.")
#SALLIST = ('Mr', 'Miss', 'Master', 'Mrs')

def name_extract(name):
    match = RE.findall(name)
    try:
        return match[0][2:-1]
    except IndexError:
        return np.NaN