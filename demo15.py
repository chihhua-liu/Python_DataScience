#demo15

import pandas as pd

df1 = pd.DataFrame({'Location': {0: 'Taipei', 1: "Hsinchu", 2: "Taichung"},
                    'Java': {0: 5, 1: 10, 2: 15},
                    'Python': {0: 2, 1: 4, 2: 6},
                    'C#': {0: 10, 1: 15, 2: 20}})
print(df1)
print("=======================================================")
print(pd.melt(df1, id_vars=['Location'], value_vars=['Java']))
print("=======================================================")
print(pd.melt(df1, id_vars=['Location'], value_vars=['Java', 'Python']))
print("=======================================================")
print(pd.melt(df1, id_vars=['Location']))