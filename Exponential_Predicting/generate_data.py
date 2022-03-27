import numpy as np
import pandas as pd

#Returns a dataframe of calculated exponential functions for given C-parameters and L-angular quantum numbers
class GenerateData:
    def __init__(self, l):
        self.l = l
        self.x = np.arange(l)
        self.c = np.arange(start=0.0001, stop=1, step = 0.0001)

    #Evaluate function for given c-values
    @staticmethod
    def func(input, arg):
        return np.e**(-input/arg)

    #Create dataset: 100,000 funcs?
    def populate(self):
        func_df = pd.DataFrame(index = self.c.tolist())
        for i in self.x:
            col = []
            for j in self.c:
                col.append(GenerateData(self.l).func(input = i, arg = j))
            func_df[i] = col
        func_df['c'] = self.c
        return func_df


#Set param value for integer L-value (angular momentum quantum number)
def data(l):
    return GenerateData(l).populate()

#%%
