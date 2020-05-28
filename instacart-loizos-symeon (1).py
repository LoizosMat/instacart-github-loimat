#!/usr/bin/env python
# coding: utf-8

# In[1]:


# For data manipulation
import pandas as pd              

# Garbage Collector to free up memory
import gc                         
gc.enable()                       # Activate 


# In[2]:


orders = pd.read_csv('orders.csv')
order_products_train = pd.read_csv('order_products__train.csv')
order_products_prior = pd.read_csv('order_products__prior.csv')
products = pd.read_csv('products.csv')
aisles = pd.read_csv('aisles.csv')
departments = pd.read_csv('departments.csv')


# In[3]:


orders = orders.loc[orders.user_id.isin(orders.user_id.drop_duplicates().sample(frac=0.1, random_state=25))] 


# In[4]:


# We convert character variables into category. 
# In Python, a categorical variable is called category and has a fixed number of different values
aisles['aisle'] = aisles['aisle'].astype('category')
departments['department'] = departments['department'].astype('category')
orders['eval_set'] = orders['eval_set'].astype('category')
products['product_name'] = products['product_name'].astype('category')


# In[5]:


#Merge the orders DF with order_products_prior by their order_id, keep only these rows with order_id that they are appear on both DFs
op = orders.merge(order_products_prior, on='order_id', how='inner')
op.head()


# In[6]:


op['order_number_back'] = op.groupby('user_id')['order_number'].transform(max) - op.order_number +1
op5 = op[op.order_number_back <= 5]
op5.head()
last_five = op5.groupby(['user_id','product_id'])[['order_id']].count()
last_five.columns = ['times_last5']
last_five = last_five.reset_index()
last_five.head(10)


# In[7]:


last_five.tail()


# In[8]:


last_five['times_last5_ratio'] = last_five.times_last5 / 5
last_five.head()


# In[9]:


max_days_since_last_order_l5 = op5.groupby(['user_id','product_id'])[['days_since_prior_order']].max()
max_days_since_last_order_l5.columns = ['max_days_since_last5'] 
max_days_since_last_order_l5 = max_days_since_last_order_l5.reset_index() 
max_days_since_last_order_l5.head()


# In[10]:


max_days_since_last_order_l5 = max_days_since_last_order_l5.fillna(0)
max_days_since_last_order_l5.head()


# In[11]:


last_five = last_five.merge(max_days_since_last_order_l5, on=['user_id','product_id'], how='left')
last_five.head()


# In[12]:


del [max_days_since_last_order_l5]


# In[13]:


max_days_since_last_order = op.groupby(['user_id','product_id'])[['days_since_prior_order']].max()
max_days_since_last_order.columns = ['max_days_since_last'] 
max_days_since_last_order = max_days_since_last_order.reset_index() 
max_days_since_last_order.head()


# In[14]:


max_days_since_last_order = max_days_since_last_order.fillna(0)
max_days_since_last_order.head()


# In[15]:


days_since_last_order = op.groupby(['user_id','product_id'])[['days_since_prior_order']].count()
days_since_last_order.columns = ['days_since_last_order'] 
days_since_last_order = days_since_last_order.reset_index() 
days_since_last_order.head()


# In[16]:


days_since_last_order = days_since_last_order.fillna(0)
days_since_last_order.head()


# In[17]:


days_last_order_max = pd.merge(days_since_last_order, max_days_since_last_order , on=['user_id', 'product_id'], how='left')
days_last_order_max.head()


# In[18]:


days_last_order_max['days_last_order_max'] = days_last_order_max.days_since_last_order / days_last_order_max.max_days_since_last
days_last_order_max.head()


# In[19]:


del [days_since_last_order, max_days_since_last_order]


# In[20]:


last_five = last_five.merge(days_last_order_max, on=['user_id','product_id'], how='left')
last_five.head()


# In[21]:


median_days_since_last_order_l5 = op5.groupby(['user_id','product_id'])[['days_since_prior_order']].median()
median_days_since_last_order_l5.columns = ['median_days_since_last5'] 
median_days_since_last_order_l5 = median_days_since_last_order_l5.reset_index() 
median_days_since_last_order_l5.head()


# In[22]:


median_days_since_last_order = op.groupby(['user_id','product_id'])[['days_since_prior_order']].median()
median_days_since_last_order.columns = ['median_days_since_last'] 
median_days_since_last_order = median_days_since_last_order.reset_index() 
median_days_since_last_order.head()


# In[23]:


median_days_since_last_order = median_days_since_last_order.merge(median_days_since_last_order_l5, on=['user_id','product_id'], how='left')
median_days_since_last_order.head()


# In[24]:


median_days_since_last_order = median_days_since_last_order.fillna(0)
median_days_since_last_order.head()


# In[25]:


del [median_days_since_last_order_l5]


# In[26]:


min_days_since_last_order_l5 = op5.groupby(['user_id','product_id'])[['days_since_prior_order']].min()
min_days_since_last_order_l5.columns = ['min_days_since_last5'] 
min_days_since_last_order_l5 = min_days_since_last_order_l5.reset_index() 
min_days_since_last_order_l5.head()


# In[27]:


min_days_since_last_order_l5 = min_days_since_last_order_l5.fillna(0)
min_days_since_last_order_l5.head()


# In[28]:


last_five = last_five.merge(min_days_since_last_order_l5, on=['user_id','product_id'], how='left')
last_five.head()


# In[29]:


del [min_days_since_last_order_l5]


# In[30]:


mean_days_since_last_order_l5 = op5.groupby(['user_id','product_id'])[['days_since_prior_order']].mean()
mean_days_since_last_order_l5.columns = ['mean_days_since_last5'] 
mean_days_since_last_order_l5 = mean_days_since_last_order_l5.reset_index() 
mean_days_since_last_order_l5.head()


# In[31]:


mean_days_since_last_order_l5 = mean_days_since_last_order_l5.fillna(0) 
mean_days_since_last_order_l5.head()


# In[32]:


last_five = last_five.merge(mean_days_since_last_order_l5, on=['user_id','product_id'], how='left')
last_five.head()


# In[33]:


del [mean_days_since_last_order_l5]


# In[34]:


last_five = last_five.merge(median_days_since_last_order, on=['user_id','product_id'], how='left')
last_five.head()


# In[35]:


del [median_days_since_last_order]


# In[36]:


avg_pos = op.groupby('product_id').filter(lambda x: x.shape[0]>30)
avg_pos = op.groupby('product_id')['add_to_cart_order'].mean().to_frame('average_position_of_a_product')
avg_pos.head()


# In[37]:


avg_pos = avg_pos.reset_index()
avg_pos.head()


# In[38]:


prr = op.groupby('product_id').filter(lambda x: x.shape[0] >40)
prr = op.groupby('product_id')['reordered'].mean().to_frame('product_reordered_ratio') #
prr.head()


# In[39]:


prr = prr.reset_index()
prr.head()


# In[40]:


prr = prr.merge(avg_pos, on='product_id', how='left')
prr.head()


# In[41]:


order_size = op.groupby(['user_id', 'order_id'])['product_id'].count().to_frame('size')
order_size = order_size.reset_index()
order_size.head()


# In[42]:


avg_os = order_size.groupby('user_id')['size'].mean().to_frame('average_order_size_for_user')
avg_os = avg_os.reset_index()
avg_os.head()


# In[43]:


last_five = last_five.merge(avg_os, on='user_id', how='left')
last_five.head()


# In[44]:


del [order_size, avg_os]


# In[45]:


aatco = op.groupby(['user_id', 'product_id'])['add_to_cart_order'].mean().to_frame('average_add_to_cart_order')
aatco = aatco.reset_index()
aatco.head()


# In[46]:


adspo = op.groupby('user_id')['days_since_prior_order'].mean().to_frame('average_days_since_prior_order')
adspo = adspo.reset_index()
adspo.head()


# In[47]:


aatco = aatco.merge(adspo, on='user_id', how='left')
aatco.head()


# In[48]:


del [adspo]


# In[49]:


last_five = last_five.merge(aatco, on=['user_id','product_id'], how='left')
last_five.head()


# In[50]:


del [aatco]


# In[51]:


user = op.groupby('user_id')['order_number'].max().to_frame('u_total_orders')
user = user.reset_index()
user.head()


# In[52]:


urr = op.groupby('user_id')['reordered'].mean().to_frame('user_reordered_ratio') #
urr = urr.reset_index()
urr.head()


# In[53]:


user = user.merge(urr, on='user_id', how='left')

del urr
gc.collect()

user.head() 


# In[54]:


uxp = op.groupby(['user_id', 'product_id'])['order_id'].count().to_frame('uxp_total_bought')
uxp.head()


# In[55]:


uxp = uxp.reset_index()
uxp.head()


# In[56]:


times = op.groupby(['user_id', 'product_id'])[['order_id']].count()
times.columns = ['Times_Bought_N']
times.head()


# In[57]:


total_orders = op.groupby('user_id')['order_number'].max().to_frame('total_orders')
total_orders.head()


# In[58]:


first_order_no = op.groupby(['user_id', 'product_id'])['order_number'].min().to_frame('first_order_number')
first_order_no  = first_order_no.reset_index()
first_order_no.head()


# In[59]:


span = pd.merge(total_orders, first_order_no, on='user_id', how='right')
span.head()


# In[60]:


span['Order_Range_D'] = span.total_orders - span.first_order_number + 1
span.head()


# In[61]:


uxp_ratio = pd.merge(times, span, on=['user_id', 'product_id'], how='left')
uxp_ratio.head()


# In[62]:


uxp_ratio['uxp_reorder_ratio'] = uxp_ratio.Times_Bought_N / uxp_ratio.Order_Range_D
uxp_ratio.head()


# In[63]:


uxp_ratio = uxp_ratio.drop(['Times_Bought_N', 'total_orders', 'first_order_number', 'Order_Range_D'], axis=1)
uxp_ratio.head()


# In[64]:


del [times, first_order_no, span]


# In[65]:


uxp = uxp.merge(uxp_ratio, on=['user_id', 'product_id'], how='left')

del uxp_ratio
uxp.head()


# In[66]:


uxp_last5 = op5.groupby(['user_id', 'product_id'])['order_id'].count().to_frame('uxp_total_bought_l5')
uxp_last5.head()


# In[67]:


uxp_last5 = uxp_last5.reset_index()
uxp_last5.head()


# In[68]:


times_l5 = op5.groupby(['user_id', 'product_id'])[['order_id']].count()
times_l5.columns = ['Times_Bought_N_l5']
times_l5.head()


# In[69]:


total_orders_l5 = op5.groupby('user_id')['order_number'].max().to_frame('total_orders_l5')
total_orders_l5.head()


# In[70]:


first_order_no_l5 = op5.groupby(['user_id', 'product_id'])['order_number'].min().to_frame('first_order_number_l5')
first_order_no_l5  = first_order_no_l5.reset_index()
first_order_no_l5.head()


# In[71]:


span_l5 = pd.merge(total_orders_l5, first_order_no_l5, on='user_id', how='right')
span_l5.head()


# In[72]:


span_l5['Order_Range_D_l5'] = span_l5.total_orders_l5 - span_l5.first_order_number_l5 + 1
span_l5.head()


# In[73]:


uxp_ratio_last5 = pd.merge(times_l5, span_l5, on=['user_id', 'product_id'], how='left')
uxp_ratio_last5.head()


# In[74]:


uxp_ratio_last5['uxp_reorder_ratio_last5'] = uxp_ratio_last5.Times_Bought_N_l5 / uxp_ratio_last5.Order_Range_D_l5
uxp_ratio_last5.head()


# In[75]:


uxp_ratio_last5 = uxp_ratio_last5.drop(['Times_Bought_N_l5', 'total_orders_l5', 'first_order_number_l5', 'Order_Range_D_l5'], axis=1)
uxp_ratio_last5.head()


# In[76]:


del [times_l5, first_order_no_l5, span_l5]


# In[77]:


uxp = uxp.merge(uxp_ratio_last5, on=['user_id', 'product_id'], how='left')


# In[78]:


del uxp_ratio_last5
uxp.head()


# In[79]:


uxp = uxp.merge(last_five, on=['user_id', 'product_id'], how='left')

del [last_five]
uxp.head()


# In[80]:


uxp = uxp.fillna(0)
uxp.head()


# In[81]:


data = uxp.merge(user, on='user_id', how='left')
data.head()


# In[82]:


data = data.merge(prr, on='product_id', how='left')
data.head()


# In[83]:


del op, user, prr, uxp
gc.collect()


# In[84]:


## First approach:
# In two steps keep only the future orders from all customers: train & test 
orders_future = orders[((orders.eval_set=='train') | (orders.eval_set=='test'))]
orders_future = orders_future[ ['user_id', 'eval_set', 'order_id'] ]
orders_future.head(10)

## Second approach (if you want to test it you have to re-run the notebook):
# In one step keep only the future orders from all customers: train & test 
#orders_future = orders.loc[((orders.eval_set=='train') | (orders.eval_set=='test')), ['user_id', 'eval_set', 'order_id'] ]
#orders_future.head(10)

## Third approach (if you want to test it you have to re-run the notebook):
# In one step exclude all the prior orders so to deal with the future orders from all customers
#orders_future = orders.loc[orders.eval_set!='prior', ['user_id', 'eval_set', 'order_id'] ]
#orders_future.head(10)


# In[85]:


data = data.merge(orders_future, on='user_id', how='left')
data.head(10)


# In[86]:


#Keep only the customers who we know what they bought in their future order
data_train = data[data.eval_set=='train'] #
data_train.head()


# In[87]:


#Get from order_products_train all the products that the train users bought bought in their future order
data_train = data_train.merge(order_products_train[['product_id','order_id', 'reordered']], on=['product_id','order_id'], how='left' )
data_train.head(15)


# In[88]:


#Where the previous merge, left a NaN value on reordered column means that the customers they haven't bought the product. We change the value on them to 0.
data_train['reordered'] = data_train['reordered'].fillna(0)
data_train.head(15)


# In[89]:


#We set user_id and product_id as the index of the DF
data_train = data_train.set_index(['user_id', 'product_id'])
data_train.head(15)


# In[90]:


#We remove all non-predictor variables
data_train = data_train.drop(['eval_set', 'order_id'], axis=1)
data_train.head(15)


# In[91]:


data_test = data[data.eval_set=='test'] #
data_test.head()


# In[92]:


#We set user_id and product_id as the index of the DF
data_test = data_test.set_index(['user_id', 'product_id']) #
data_test.head()


# In[93]:


#We remove all non-predictor variables
data_test = data_test.drop(['eval_set','order_id'], axis=1)
#Check if the data_test DF, has the same number of columns as the data_train DF, excluding the response variable
data_test.head()


# In[94]:


import sys
get_ipython().system('{sys.executable} -m pip install xgboost')


# In[95]:


# TRAIN FULL 
###########################
## IMPORT REQUIRED PACKAGES
###########################
import xgboost as xgb

##########################################
## SPLIT DF TO: X_train, y_train (axis=1)
##########################################
X_train, y_train = data_train.drop('reordered', axis=1), data_train.reordered

########################################
## SET BOOSTER'S PARAMETERS
########################################
parameters = {'eval_metric':'logloss', 
              'max_depth':'5', 
              'colsample_bytree':'0.6',
              'subsample':'0.8',
             }

########################################
## INSTANTIATE XGBClassifier()
########################################
xgbc = xgb.XGBClassifier(objective='binary:logistic', parameters=parameters, num_boost_round=10)

########################################
## TRAIN MODEL
########################################
model = xgbc.fit(X_train, y_train)

##################################
# FEATURE IMPORTANCE - GRAPHICAL
##################################
xgb.plot_importance(model)


# In[96]:


model.get_xgb_params()


# In[97]:


###########################
## DISABLE WARNINGS
###########################
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

###########################
## IMPORT REQUIRED PACKAGES
###########################
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

####################################
## SET BOOSTER'S RANGE OF PARAMETERS
# IMPORTANT NOTICE: Fine-tuning an XGBoost model may be a computational prohibitive process with a regular computer or a Kaggle kernel. 
# Be cautious what parameters you enter in paramiGrid section.
# More paremeters means that GridSearch will create and evaluate more models.
####################################    
paramGrid = {"max_depth":[5,10],
            "colsample_bytree":[0.3,0.4]}  

########################################
## INSTANTIATE XGBClassifier()
########################################
xgbc = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', num_boost_round=10)

##############################################
## DEFINE HOW TO TRAIN THE DIFFERENT MODELS
#############################################
gridsearch = GridSearchCV(xgbc, paramGrid, cv=3, verbose=2, n_jobs=1)

################################################################
## TRAIN THE MODELS
### - with the combinations of different parameters
### - here is where GridSearch will be exeucuted
#################################################################
model = gridsearch.fit(X_train, y_train)

##################################
## OUTPUT(S)
##################################
# Print the best parameters
print("The best parameters are: /n",  gridsearch.best_params_)

# Store the model for prediction (chapter 5)
model = gridsearch.best_estimator_

# Delete X_train , y_train
del [X_train, y_train]


# In[98]:


model.get_params()


# In[99]:


# Predict values for test data with our model from chapter 5 - the results are saved as a Python array
test_pred = model.predict(data_test).astype(int)
test_pred[0:20] #display the first 20 predictions of the numpy array


# In[100]:


## OR set a custom threshold (in this problem, 0.21 yields the best prediction)
test_pred = (model.predict_proba(data_test)[:,1] >= 0.21).astype(int)
test_pred[0:20] #display the first 20 predictions of the numpy array


# In[101]:


#Save the prediction in a new column in the data_test DF
data_test['prediction'] = test_pred
data_test.head()


# In[102]:


#Reset the index
final = data_test.reset_index()
#Keep only the required columns to create our submission file (Chapter 6)
final = final[['product_id', 'user_id', 'prediction']]

gc.collect()
final.head()


# In[103]:


orders_test = orders.loc[orders.eval_set=='test',("user_id", "order_id") ]
orders_test.head()


# In[104]:


final = final.merge(orders_test, on='user_id', how='left')
final.head()


# In[105]:


#remove user_id column
final = final.drop('user_id', axis=1)
#convert product_id as integer
final['product_id'] = final.product_id.astype(int)

#Remove all unnecessary objects
del orders
del orders_test
gc.collect()

final.head()


# In[106]:


d = dict()
for row in final.itertuples():
    if row.prediction== 1:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)

for order in final.order_id:
    if order not in d:
        d[order] = 'None'
        
gc.collect()

#We now check how the dictionary were populated (open hidden output)
d


# In[107]:


#Convert the dictionary into a DataFrame
sub = pd.DataFrame.from_dict(d, orient='index')

#Reset index
sub.reset_index(inplace=True)
#Set column names
sub.columns = ['order_id', 'products']

sub.head()


# In[108]:


#Check if sub file has 75000 predictions
sub.shape[0]
print(sub.shape[0]==75000)


# In[109]:


sub.to_csv('sub.csv', index=False)

