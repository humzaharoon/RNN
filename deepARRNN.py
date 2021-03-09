#!/usr/bin/env python
# coding: utf-8

# In[133]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.model.forecast import SampleForecast


# In[134]:


df=pd.read_csv('data_share.csv', sep=',', index_col = 0, parse_dates=True)


# In[135]:


df['y'] = pd.to_numeric(df["y"], downcast="float")


# In[ ]:





# In[136]:


df


# In[137]:


df.T.iloc[:, 234:247]


# In[ ]:





# In[138]:


df_input = df.reset_index(drop=True).T.reset_index()


# In[139]:


df_input.iloc[:, 233]
ts_code=df_input['index'].astype('category').cat.codes.values
ts_code.reshape(-1,1)
df_test1 = df_input.iloc[:, 235:247]
df_test1


# In[140]:


df_training = df_input.iloc[:, 1:233].values
df_test = df_input.iloc[:, 234:247].values

df_training


# In[141]:


freq='3M'
start_train = pd.Timestamp('1959-09-01', freq=freq)
start_test = pd.Timestamp('2018-03-01', freq=freq)
future_forecast = pd.Timestamp('')
prediction_length = 6


# In[146]:


estimator = DeepAREstimator(freq=freq,
                           prediction_length = prediction_length,
                           context_length = 245,
                           use_feat_static_cat=True,
                           cardinality=[11],
                           num_layers=2,
                           num_cells=32,
                           cell_type='lstm',
                           lags_seq = [5], 
                           dropout_rate = 0.3,  
                           trainer=Trainer(epochs=50, learning_rate=1E-16))


# In[147]:


from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName


# In[148]:


train_ds = ListDataset([
    {
        FieldName.TARGET: target,
        FieldName.START: start_train,
        FieldName.FEAT_STATIC_CAT:fsc
    }
    for (target, fsc) in zip(df_training, ts_code.reshape(-1, 1))
], freq=freq)
test_ds = ListDataset([
    {
        FieldName.TARGET: target,
        FieldName.START: start_test,
        FieldName.FEAT_STATIC_CAT:fsc
        
    }
    for (target, fsc) in zip(df_test, ts_code.reshape(-1, 1))
], freq=freq)


# In[149]:


predictor = estimator.train(training_data=train_ds)


# In[150]:


from gluonts.evaluation.backtest import make_evaluation_predictions


# In[151]:


forecast_it, ts = make_evaluation_predictions(
                    dataset=test_ds,
                    predictor=predictor,
                    num_samples=100)


# In[165]:


forecast_it


# In[152]:


ts


# In[153]:


from tqdm.autonotebook import tqdm


# In[154]:


tss = list(tqdm(ts, total=len(df_test)))
forecast = list(tqdm(forecast_it, total=len(df_test)))


# In[155]:


from gluonts.evaluation import Evaluator


# In[156]:


evaluator = Evaluator(quantiles = [0.25, 0.5, 0.6])
agg_metrics, item_metrics = evaluator(iter(tss), iter(forecast), num_series = len(df_test))


# In[157]:


item_metrics


# In[158]:


tss[10].head()


# In[167]:


next(iter(forecast))


# In[130]:





# In[132]:





# In[ ]:




