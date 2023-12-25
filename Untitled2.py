#!/usr/bin/env python
# coding: utf-8

# In[119]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

mod = KNeighborsRegressor().fit(X,y)

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model",KNeighborsRegressor(n_neighbors=1))])
#pipe.get_params()



 
pipe.fit(X,y)
                    
    

Pipeline(steps=[('scale',StandardScaler()),('model',KNeighborsRegressor())])



from sklearn.datasets import load_diabetes

X,y = load_diabetes(return_X_y=True)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

mod = LinearRegression()

mod.fit(X,y)

mod.predict(X)

pred = pipe.predict(X
                  )


# In[118]:


plt.scatter(pred, y)


# In[120]:


mod = GridSearchCV(estimator=pipe,
                   param_grid={'model__n_neighbors':[1,2,3,4,5,6,7,8,9,10] },
                   cv=3)


# In[121]:


mod.fit(X,y); 
pd.DataFrame(mod.cv_results_)


# In[124]:


print(load_diabetes()['DESCR'])


# In[ ]:




