#!/usr/bin/env python
# coding: utf-8

# In[1]:


# For data manipulation
import pandas as pd            
import time

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


orders = orders.loc[orders.user_id.isin(orders.user_id.drop_duplicates().sample(frac=0.05, random_state=25))] 


# In[4]:


# We convert character variables into category. 
# In Python, a categorical variable is called category and has a fixed number of different values
aisles['aisle'] = aisles['aisle'].astype('category')
departments['department'] = departments['department'].astype('category')
orders['eval_set'] = orders['eval_set'].astype('category')
products['product_name'] = products['product_name'].astype('category')


# In[5]:


orders.sort_values('order_hour_of_day', inplace=True)
orders.drop_duplicates(inplace=True)
orders.reset_index(drop=True, inplace=True)


# In[6]:


def timezone(s):
    if s < 6:
        return '0'
    elif s < 12:
        return '1'
    elif s < 18:
        return '2'
    else:
        return '3'


# In[7]:


orders['timezone'] = orders.order_hour_of_day.map(timezone)
orders.head()


# In[8]:


orders['timezone'] = orders['timezone'].astype('category')


# In[9]:


#Merge the orders DF with order_products_prior by their order_id, keep only these rows with order_id that they are appear on both DFs
op = orders.merge(order_products_prior, on='order_id', how='inner')
op.head()


# In[10]:


opad = op.merge(products, on='product_id', how='left')
opad.head()


# In[11]:


aipr = opad.groupby(['aisle_id','product_id'])['order_id'].count().to_frame('aisle_total_orders_of_each_product')
aipr = aipr.reset_index()
aipr.head()


# In[12]:


ais = opad.groupby('aisle_id')['order_id'].count().to_frame('aisle_total_orders')
ais = ais.reset_index()
ais.head()


# In[13]:


aire = opad.groupby('aisle_id')['reordered'].mean().to_frame('aisle_reordered_ratio') #
aire = aire.reset_index()
aire.head()


# In[14]:


aisles = pd.merge(aipr, ais, on='aisle_id', how='inner')
aisles.head()


# In[15]:


avg_ais = opad.groupby('aisle_id')['add_to_cart_order'].mean().to_frame('average_position_of_an_aisle')
avg_ais.head()


# In[16]:


avg_ais = avg_ais.reset_index()
avg_ais.head()


# In[17]:


aisles = aisles.merge(avg_ais, on='aisle_id', how='left')
aisles.head()


# In[18]:


aisles = aisles.merge(aire, on='aisle_id', how='left')
aisles.head()


# In[19]:


depr = opad.groupby(['department_id','product_id'])['order_id'].count().to_frame('department_total_orders_for_each_product')
depr = depr.reset_index()
depr.head()


# In[20]:


dep = opad.groupby('department_id')['order_id'].count().to_frame('department_total_orders')
dep = dep.reset_index()
dep.head()


# In[21]:


dere = opad.groupby('department_id')['reordered'].mean().to_frame('department_reordered_ratio') #
dere = dere.reset_index()
dere.head()


# In[22]:


departments = pd.merge(depr, dep, on='department_id', how='inner')
departments.head()


# In[23]:


avg_dep = opad.groupby('department_id')['add_to_cart_order'].mean().to_frame('average_position_of_a_department')
avg_dep = avg_dep.reset_index()
avg_dep.head()


# In[24]:


departments = departments.merge(avg_dep, on='department_id', how='left')
departments.head()


# In[25]:


departments = departments.merge(dere, on='department_id', how='left')
departments.head()


# In[26]:


ad = pd.merge(aisles, departments, on='product_id', how='inner')
ad.head()


# In[27]:


del [opad, aipr, depr, aisles, departments]


# In[28]:


op['order_number_back'] = op.groupby('user_id')['order_number'].transform(max) - op.order_number +1
op5 = op[op.order_number_back <= 5]
op5.head()
last_five = op5.groupby(['user_id','product_id'])[['order_id']].count()
last_five.columns = ['times_last5']
last_five = last_five.reset_index()
last_five.head(10)


# In[29]:


last_five.tail()


# In[30]:


last_five['times_last5_ratio'] = last_five.times_last5 / 5
last_five.head()


# In[31]:


max_days_since_last_order_l5 = op5.groupby(['user_id','product_id'])[['days_since_prior_order']].max()
max_days_since_last_order_l5.columns = ['max_days_since_last5'] 
max_days_since_last_order_l5 = max_days_since_last_order_l5.reset_index() 
max_days_since_last_order_l5.head()


# In[32]:


max_days_since_last_order_l5 = max_days_since_last_order_l5.fillna(0)
max_days_since_last_order_l5.head()


# In[33]:


last_five = last_five.merge(max_days_since_last_order_l5, on=['user_id','product_id'], how='left')
last_five.head()


# In[34]:


del [max_days_since_last_order_l5]


# In[35]:


max_days_since_last_order = op.groupby(['user_id','product_id'])[['days_since_prior_order']].max()
max_days_since_last_order.columns = ['max_days_since_last'] 
max_days_since_last_order = max_days_since_last_order.reset_index() 
max_days_since_last_order.head()


# In[36]:


max_days_since_last_order = max_days_since_last_order.fillna(0)
max_days_since_last_order.head()


# In[37]:


days_since_last_order = op.groupby(['user_id','product_id'])[['days_since_prior_order']].count()
days_since_last_order.columns = ['days_since_last_order'] 
days_since_last_order = days_since_last_order.reset_index() 
days_since_last_order.head()


# In[38]:


days_since_last_order = days_since_last_order.fillna(0)
days_since_last_order.head()


# In[39]:


days_last_order_max = pd.merge(days_since_last_order, max_days_since_last_order , on=['user_id', 'product_id'], how='left')
days_last_order_max.head()


# In[40]:


days_last_order_max['days_last_order_max'] = days_last_order_max.days_since_last_order / days_last_order_max.max_days_since_last
days_last_order_max.head()


# In[41]:


del [days_since_last_order, max_days_since_last_order]


# In[42]:


last_five = last_five.merge(days_last_order_max, on=['user_id','product_id'], how='left')
last_five.head()


# In[43]:


median_days_since_last_order_l5 = op5.groupby(['user_id','product_id'])[['days_since_prior_order']].median()
median_days_since_last_order_l5.columns = ['median_days_since_last5'] 
median_days_since_last_order_l5 = median_days_since_last_order_l5.reset_index() 
median_days_since_last_order_l5.head()


# In[44]:


median_days_since_last_order = op.groupby(['user_id','product_id'])[['days_since_prior_order']].median()
median_days_since_last_order.columns = ['median_days_since_last'] 
median_days_since_last_order = median_days_since_last_order.reset_index() 
median_days_since_last_order.head()


# In[45]:


median_days_since_last_order = median_days_since_last_order.merge(median_days_since_last_order_l5, on=['user_id','product_id'], how='left')
median_days_since_last_order.head()


# In[46]:


median_days_since_last_order = median_days_since_last_order.fillna(0)
median_days_since_last_order.head()


# In[47]:


del [median_days_since_last_order_l5]


# In[48]:


min_days_since_last_order_l5 = op5.groupby(['user_id','product_id'])[['days_since_prior_order']].min()
min_days_since_last_order_l5.columns = ['min_days_since_last5'] 
min_days_since_last_order_l5 = min_days_since_last_order_l5.reset_index() 
min_days_since_last_order_l5.head()


# In[49]:


min_days_since_last_order_l5 = min_days_since_last_order_l5.fillna(0)
min_days_since_last_order_l5.head()


# In[50]:


last_five = last_five.merge(min_days_since_last_order_l5, on=['user_id','product_id'], how='left')
last_five.head()


# In[51]:


del [min_days_since_last_order_l5]


# In[52]:


mean_days_since_last_order_l5 = op5.groupby(['user_id','product_id'])[['days_since_prior_order']].mean()
mean_days_since_last_order_l5.columns = ['mean_days_since_last5'] 
mean_days_since_last_order_l5 = mean_days_since_last_order_l5.reset_index() 
mean_days_since_last_order_l5.head()


# In[53]:


mean_days_since_last_order_l5 = mean_days_since_last_order_l5.fillna(0) 
mean_days_since_last_order_l5.head()


# In[54]:


last_five = last_five.merge(mean_days_since_last_order_l5, on=['user_id','product_id'], how='left')
last_five.head()


# In[55]:


del [mean_days_since_last_order_l5]


# In[56]:


last_five = last_five.merge(median_days_since_last_order, on=['user_id','product_id'], how='left')
last_five.head()


# In[57]:


del [median_days_since_last_order]


# In[58]:


avg_pos = op.groupby('product_id').filter(lambda x: x.shape[0]>30)
avg_pos = op.groupby('product_id')['add_to_cart_order'].mean().to_frame('average_position_of_a_product')
avg_pos.head()


# In[59]:


avg_pos = avg_pos.reset_index()
avg_pos.head()


# In[60]:


prd = op.groupby('product_id')['order_id'].count().to_frame('p_total_purchases')
prd.head()


# In[61]:


prd = prd.reset_index()
prd.head()


# In[62]:


prr = op.groupby('product_id').filter(lambda x: x.shape[0] >40)
prr = op.groupby('product_id')['reordered'].mean().to_frame('product_reordered_ratio') #
prr.head()


# In[63]:


prr = prr.reset_index()
prr.head()


# In[64]:


prr = prr.merge(prd, on='product_id', how='right')


# In[65]:


prr = prr.merge(avg_pos, on='product_id', how='left')
prr.head()


# In[66]:


prr = prr.merge(ad, on='product_id', how='left')
prr.head()


# In[67]:


order_size = op.groupby(['user_id', 'order_id'])['product_id'].count().to_frame('size')
order_size = order_size.reset_index()
order_size.head()


# In[68]:


avg_os = order_size.groupby('user_id')['size'].mean().to_frame('average_order_size_for_user')
avg_os = avg_os.reset_index()
avg_os.head()


# In[69]:


last_five = last_five.merge(avg_os, on='user_id', how='left')
last_five.head()


# In[70]:


del [order_size, avg_os]


# In[71]:


aatco = op.groupby(['user_id', 'product_id'])['add_to_cart_order'].mean().to_frame('average_add_to_cart_order')
aatco = aatco.reset_index()
aatco.head()


# In[72]:


adspo = op.groupby('user_id')['days_since_prior_order'].mean().to_frame('average_days_since_prior_order')
adspo = adspo.reset_index()
adspo.head()


# In[73]:


aatco = aatco.merge(adspo, on='user_id', how='left')
aatco.head()


# In[74]:


del [adspo]


# In[75]:


#1.6 num_distinct_product the number of distinct product of user
def f1(x):
    return len(set(x))

num_distinct_prodct = op.groupby('user_id')['product_id'].apply(f1).to_frame('num_distinct_product')
num_distinct_prodct = num_distinct_prodct.reset_index()
num_distinct_prodct.head()


# In[76]:


aatco = aatco.merge(num_distinct_prodct, on='user_id', how='left')
aatco.head()


# In[77]:


last_five = last_five.merge(aatco, on=['user_id','product_id'], how='left')
last_five.head()


# In[78]:


del [aatco]


# In[79]:


user = op.groupby('user_id')['order_number'].max().to_frame('u_total_orders')
user = user.reset_index()
user.head()


# In[80]:


urr = op.groupby('user_id')['reordered'].mean().to_frame('user_reordered_ratio') #
urr = urr.reset_index()
urr.head()


# In[81]:


user = user.merge(urr, on='user_id', how='left')

del urr
gc.collect()

user.head() 


# In[82]:


utz = op.groupby(['user_id','timezone'])['order_hour_of_day'].count().to_frame('user_timezone')
utz = utz.reset_index()
utz.head()


ptz = op.groupby(['product_id','timezone'])['order_hour_of_day'].count().to_frame('product_timezone')
ptz = ptz.reset_index()
ptz.head()

utz = utz.merge(ptz, on='timezone', how='left')
utz.head()

otz = op.groupby(['order_id','timezone'])['order_hour_of_day'].count().to_frame('order_timezone')
otz = otz.reset_index()
otz.head()

utz = utz.merge(otz, on='timezone', how='left')
utz.head()
# In[83]:


user = user.merge(utz, on='user_id', how='left')
utz.head()


# In[84]:


del [utz]


# In[85]:


uxp = op.groupby(['user_id', 'product_id'])['order_id'].count().to_frame('uxp_total_bought')
uxp.head()


# In[86]:


uxp = uxp.reset_index()
uxp.head()


# In[87]:


times = op.groupby(['user_id', 'product_id'])[['order_id']].count()
times.columns = ['Times_Bought_N']
times.head()


# In[88]:


total_orders = op.groupby('user_id')['order_number'].max().to_frame('total_orders')
total_orders.head()


# In[89]:


first_order_no = op.groupby(['user_id', 'product_id'])['order_number'].min().to_frame('first_order_number')
first_order_no  = first_order_no.reset_index()
first_order_no.head()


# In[90]:


span = pd.merge(total_orders, first_order_no, on='user_id', how='right')
span.head()


# In[91]:


span['Order_Range_D'] = span.total_orders - span.first_order_number + 1
span.head()


# In[92]:


uxp_ratio = pd.merge(times, span, on=['user_id', 'product_id'], how='left')
uxp_ratio.head()


# In[93]:


uxp_ratio['uxp_reorder_ratio'] = uxp_ratio.Times_Bought_N / uxp_ratio.Order_Range_D
uxp_ratio.head()


# In[94]:


uxp_ratio = uxp_ratio.drop(['Times_Bought_N', 'total_orders', 'first_order_number', 'Order_Range_D'], axis=1)
uxp_ratio.head()


# In[95]:


del [times, first_order_no, span]


# In[96]:


uxp = uxp.merge(uxp_ratio, on=['user_id', 'product_id'], how='left')

del uxp_ratio
uxp.head()


# In[97]:


uxp_last5 = op5.groupby(['user_id', 'product_id'])['order_id'].count().to_frame('uxp_total_bought_l5')
uxp_last5.head()


# In[98]:


uxp_last5 = uxp_last5.reset_index()
uxp_last5.head()


# In[99]:


times_l5 = op5.groupby(['user_id', 'product_id'])[['order_id']].count()
times_l5.columns = ['Times_Bought_N_l5']
times_l5.head()


# In[100]:


total_orders_l5 = op5.groupby('user_id')['order_number'].max().to_frame('total_orders_l5')
total_orders_l5.head()


# In[101]:


first_order_no_l5 = op5.groupby(['user_id', 'product_id'])['order_number'].min().to_frame('first_order_number_l5')
first_order_no_l5  = first_order_no_l5.reset_index()
first_order_no_l5.head()


# In[102]:


span_l5 = pd.merge(total_orders_l5, first_order_no_l5, on='user_id', how='right')
span_l5.head()


# In[103]:


span_l5['Order_Range_D_l5'] = span_l5.total_orders_l5 - span_l5.first_order_number_l5 + 1
span_l5.head()


# In[104]:


uxp_ratio_last5 = pd.merge(times_l5, span_l5, on=['user_id', 'product_id'], how='left')
uxp_ratio_last5.head()


# In[105]:


uxp_ratio_last5['uxp_reorder_ratio_last5'] = uxp_ratio_last5.Times_Bought_N_l5 / uxp_ratio_last5.Order_Range_D_l5
uxp_ratio_last5.head()


# In[106]:


uxp_ratio_last5 = uxp_ratio_last5.drop(['Times_Bought_N_l5', 'total_orders_l5', 'first_order_number_l5', 'Order_Range_D_l5'], axis=1)
uxp_ratio_last5.head()


# In[107]:


del [times_l5, first_order_no_l5, span_l5]


# In[108]:


uxp = uxp.merge(uxp_ratio_last5, on=['user_id', 'product_id'], how='left')


# In[109]:


del uxp_ratio_last5
uxp.head()


# In[110]:


uxp = uxp.merge(last_five, on=['user_id', 'product_id'], how='left')

del [last_five]
uxp.head()


# In[111]:


uxp = uxp.fillna(0)
uxp.head()


# In[112]:


data = uxp.merge(user, on='user_id', how='left')
data.head()


# In[113]:


data = data.merge(prr, on='product_id', how='left')
data.head()


# In[114]:


del op, op5, user, prr, uxp
gc.collect()


# In[115]:


## First approach:
# In two steps keep only the future orders from all customers: train & test 
orders_future = orders[((orders.eval_set=='train') | (orders.eval_set=='test'))]
orders_future = orders_future[ ['user_id', 'eval_set', 'order_id'] ]
orders_future.head(10)


# In[116]:


data = data.merge(orders_future, on='user_id', how='left')
data.head(10)


# In[117]:


#Keep only the customers who we know what they bought in their future order
data_train = data[data.eval_set=='train'] #
data_train.head()


# In[118]:


#Get from order_products_train all the products that the train users bought bought in their future order
data_train = data_train.merge(order_products_train[['product_id','order_id', 'reordered']], on=['product_id','order_id'], how='left' )
data_train.head(15)


# In[119]:


#Where the previous merge, left a NaN value on reordered column means that the customers they haven't bought the product. We change the value on them to 0.
data_train['reordered'] = data_train['reordered'].fillna(0)
data_train.head(15)


# In[120]:


#We set user_id and product_id as the index of the DF
data_train = data_train.set_index(['user_id', 'product_id'])
data_train.head(15)


# In[121]:


#We remove all non-predictor variables
data_train = data_train.drop(['eval_set', 'order_id'], axis=1)
data_train.head(15)


# In[122]:


data_test = data[data.eval_set=='test'] #
data_test.head()


# In[123]:


#We set user_id and product_id as the index of the DF
data_test = data_test.set_index(['user_id', 'product_id']) #
data_test.head()


# In[124]:


#We remove all non-predictor variables
data_test = data_test.drop(['eval_set','order_id'], axis=1)
#Check if the data_test DF, has the same number of columns as the data_train DF, excluding the response variable
data_test.head()


# In[126]:


# In[128]:


# TRAIN FULL 
###########################
## IMPORT REQUIRED PACKAGES
###########################
import xgboost as xgb


##########################################
## SPLIT DF TO: X_train, y_train (axis=1)
##########################################
X_train, y_train = data_train.drop('reordered', axis=1), data_train.reordered

from sklearn import preprocessing

lbl = preprocessing.LabelEncoder()
X_train['timezone'] = lbl.fit_transform(X_train['timezone'].astype(str))
########################################
########################################
## SET BOOSTER'S PARAMETERS
########################################
parameters = {'eval_metric':'logloss', 
              'max_depth':'8', 
              'colsample_bytree':'0.9',
              'subsample':'1',
              'min_child_weight':'1'
             }

########################################
## INSTANTIATE XGBClassifier()
########################################
xgbc = xgb.XGBClassifier(objective='binary:logistic', parameters=parameters, num_boost_round=10, gpu_id=0, tree_method= 'gpu_hist')

########################################
## TRAIN MODEL
#########################################
model = xgbc.fit(X_train, y_train)


model.get_xgb_params()




# Predict values for test data with our model from chapter 5 - the results are saved as a Python array
test_pred = model.predict(data_test).astype(int)
test_pred[0:20] #display the first 20 predictions of the numpy array


# In[ ]:


## OR set a custom threshold (in this problem, 0.21 yields the best prediction)
test_pred = (model.predict_proba(data_test)[:,1] >= 0.21).astype(int)
test_pred[0:20] #display the first 20 predictions of the numpy array


# In[ ]:


#Save the prediction in a new column in the data_test DF
data_test['prediction'] = test_pred
data_test.head()


# In[ ]:


#Reset the index
final = data_test.reset_index()
#Keep only the required columns to create our submission file (Chapter 6)
final = final[['product_id', 'user_id', 'prediction']]

gc.collect()
final.head()


# In[ ]:


orders_test = orders.loc[orders.eval_set=='test',("user_id", "order_id") ]
orders_test.head()


# In[ ]:


final = final.merge(orders_test, on='user_id', how='left')
final.head()


# In[ ]:


#remove user_id column
final = final.drop('user_id', axis=1)
#convert product_id as integer
final['product_id'] = final.product_id.astype(int)

#Remove all unnecessary objects
del orders
del orders_test
gc.collect()

final.head()


# In[ ]:


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


# In[ ]:


#Convert the dictionary into a DataFrame
sub = pd.DataFrame.from_dict(d, orient='index')

#Reset index
sub.reset_index(inplace=True)
#Set column names
sub.columns = ['order_id', 'products']

sub.head()


# In[ ]:


#Check if sub file has 75000 predictions
sub.shape[0]
print(sub.shape[0]==75000)


# In[ ]:


sub.to_csv('sub.csv', index=False)

