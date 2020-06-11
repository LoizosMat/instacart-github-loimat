#!/usr/bin/env python
# coding: utf-8

# In[207]:


# For data manipulation
import pandas as pd              

# Garbage Collector to free up memory
import gc                         
gc.enable()                       # Activate 


# In[208]:


orders = pd.read_csv('orders.csv')
order_products_train = pd.read_csv('order_products__train.csv')
order_products_prior = pd.read_csv('order_products__prior.csv')
products = pd.read_csv('products.csv')
aisles = pd.read_csv('aisles.csv')
departments = pd.read_csv('departments.csv')




# In[210]:


# We convert character variables into category. 
# In Python, a categorical variable is called category and has a fixed number of different values
aisles['aisle'] = aisles['aisle'].astype('category')
departments['department'] = departments['department'].astype('category')
orders['eval_set'] = orders['eval_set'].astype('category')
products['product_name'] = products['product_name'].astype('category')


# In[212]:


orders.sort_values('order_hour_of_day', inplace=True)
orders.drop_duplicates(inplace=True)
orders.reset_index(drop=True, inplace=True)


# In[213]:


def timezone(s):
    if s < 6:
        return 'midnight'
    elif s < 12:
        return 'morning'
    elif s < 18:
        return 'noon'
    else:
        return 'night'


# In[214]:


orders['timezone'] = orders.order_hour_of_day.map(timezone)
orders.head()


# In[216]:


#Merge the orders DF with order_products_prior by their order_id, keep only these rows with order_id that they are appear on both DFs
op = orders.merge(order_products_prior, on='order_id', how='inner')
op.head()


# In[96]:


opad = op.merge(products, on='product_id', how='left')
opad.head()


# In[97]:


aipr = opad.groupby(['aisle_id','product_id'])['order_id'].count().to_frame('aisle_total_orders_of_each_product')
aipr = aipr.reset_index()
aipr.head()


# In[98]:


ais = opad.groupby('aisle_id')['order_id'].count().to_frame('aisle_total_orders')
ais = ais.reset_index()
ais.head()


# In[99]:


aire = opad.groupby('aisle_id')['reordered'].mean().to_frame('aisle_reordered_ratio') #
aire = aire.reset_index()
aire.head()


# In[100]:


aisles = pd.merge(aipr, ais, on='aisle_id', how='inner')
aisles.head()


# In[101]:


avg_ais = opad.groupby('aisle_id')['add_to_cart_order'].mean().to_frame('average_position_of_an_aisle')
avg_ais.head()


# In[102]:


avg_ais = avg_ais.reset_index()
avg_ais.head()


# In[103]:


aisles = aisles.merge(avg_ais, on='aisle_id', how='left')
aisles.head()


# In[104]:


aisles = aisles.merge(aire, on='aisle_id', how='left')
aisles.head()


# In[75]:


ais_dow = opad.groupby(['aisle_id', 'order_dow'])['order_id'].count().to_frame('aisle_orders_for_each_day')
ais_dow = ais_dow.reset_index()
ais_dow.head()



# In[78]:


aisles = aisles.merge(ais_dow, on='aisle_id', how='left')
aisles.head()


# In[105]:


depr = opad.groupby(['department_id','product_id'])['order_id'].count().to_frame('department_total_orders_for_each_product')
depr = depr.reset_index()
depr.head()


# In[106]:


dep = opad.groupby('department_id')['order_id'].count().to_frame('department_total_orders')
dep = dep.reset_index()
dep.head()


# In[107]:


dere = opad.groupby('department_id')['reordered'].mean().to_frame('department_reordered_ratio') #
dere = dere.reset_index()
dere.head()


# In[108]:


departments = pd.merge(depr, dep, on='department_id', how='inner')
departments.head()


# In[109]:


avg_dep = opad.groupby('department_id')['add_to_cart_order'].mean().to_frame('average_position_of_a_department')
avg_dep = avg_dep.reset_index()
avg_dep.head()


# In[110]:


departments = departments.merge(avg_dep, on='department_id', how='left')
departments.head()


# In[111]:


departments = departments.merge(dere, on='department_id', how='left')
departments.head()


# In[86]:


dep_dow = opad.groupby(['department_id', 'order_dow'])['order_id'].count().to_frame('department_orders_for_each_day')
dep_dow = dep_dow.reset_index()
dep_dow.head()


# In[89]:


departments = departments.merge(dep_dow, on='department_id', how='left')
departments.head()


# In[112]:


ad = pd.merge(aisles, departments, on='product_id', how='inner')
ad.head()


# In[114]:


del [opad, aipr, depr, aisles, departments]


# In[118]:


op['order_number_back'] = op.groupby('user_id')['order_number'].transform(max) - op.order_number +1
op5 = op[op.order_number_back <= 5]
op5.head()
last_five = op5.groupby(['user_id','product_id'])[['order_id']].count()
last_five.columns = ['times_last5']
last_five = last_five.reset_index()
last_five.head(10)


# In[119]:


last_five.tail()


# In[120]:


last_five['times_last5_ratio'] = last_five.times_last5 / 5
last_five.head()


# In[121]:


max_days_since_last_order_l5 = op5.groupby(['user_id','product_id'])[['days_since_prior_order']].max()
max_days_since_last_order_l5.columns = ['max_days_since_last5'] 
max_days_since_last_order_l5 = max_days_since_last_order_l5.reset_index() 
max_days_since_last_order_l5.head()


# In[122]:


max_days_since_last_order_l5 = max_days_since_last_order_l5.fillna(0)
max_days_since_last_order_l5.head()


# In[123]:


last_five = last_five.merge(max_days_since_last_order_l5, on=['user_id','product_id'], how='left')
last_five.head()


# In[124]:


del [max_days_since_last_order_l5]


# In[125]:


max_days_since_last_order = op.groupby(['user_id','product_id'])[['days_since_prior_order']].max()
max_days_since_last_order.columns = ['max_days_since_last'] 
max_days_since_last_order = max_days_since_last_order.reset_index() 
max_days_since_last_order.head()


# In[126]:


max_days_since_last_order = max_days_since_last_order.fillna(0)
max_days_since_last_order.head()


# In[127]:


days_since_last_order = op.groupby(['user_id','product_id'])[['days_since_prior_order']].count()
days_since_last_order.columns = ['days_since_last_order'] 
days_since_last_order = days_since_last_order.reset_index() 
days_since_last_order.head()


# In[128]:


days_since_last_order = days_since_last_order.fillna(0)
days_since_last_order.head()


# In[129]:


days_last_order_max = pd.merge(days_since_last_order, max_days_since_last_order , on=['user_id', 'product_id'], how='left')
days_last_order_max.head()


# In[130]:


days_last_order_max['days_last_order_max'] = days_last_order_max.days_since_last_order / days_last_order_max.max_days_since_last
days_last_order_max.head()


# In[131]:


del [days_since_last_order, max_days_since_last_order]


# In[132]:


last_five = last_five.merge(days_last_order_max, on=['user_id','product_id'], how='left')
last_five.head()


# In[133]:


median_days_since_last_order_l5 = op5.groupby(['user_id','product_id'])[['days_since_prior_order']].median()
median_days_since_last_order_l5.columns = ['median_days_since_last5'] 
median_days_since_last_order_l5 = median_days_since_last_order_l5.reset_index() 
median_days_since_last_order_l5.head()


# In[134]:


median_days_since_last_order = op.groupby(['user_id','product_id'])[['days_since_prior_order']].median()
median_days_since_last_order.columns = ['median_days_since_last'] 
median_days_since_last_order = median_days_since_last_order.reset_index() 
median_days_since_last_order.head()


# In[135]:


median_days_since_last_order = median_days_since_last_order.merge(median_days_since_last_order_l5, on=['user_id','product_id'], how='left')
median_days_since_last_order.head()


# In[136]:


median_days_since_last_order = median_days_since_last_order.fillna(0)
median_days_since_last_order.head()


# In[137]:


del [median_days_since_last_order_l5]


# In[138]:


min_days_since_last_order_l5 = op5.groupby(['user_id','product_id'])[['days_since_prior_order']].min()
min_days_since_last_order_l5.columns = ['min_days_since_last5'] 
min_days_since_last_order_l5 = min_days_since_last_order_l5.reset_index() 
min_days_since_last_order_l5.head()


# In[139]:


min_days_since_last_order_l5 = min_days_since_last_order_l5.fillna(0)
min_days_since_last_order_l5.head()


# In[140]:


last_five = last_five.merge(min_days_since_last_order_l5, on=['user_id','product_id'], how='left')
last_five.head()


# In[141]:


del [min_days_since_last_order_l5]


# In[142]:


mean_days_since_last_order_l5 = op5.groupby(['user_id','product_id'])[['days_since_prior_order']].mean()
mean_days_since_last_order_l5.columns = ['mean_days_since_last5'] 
mean_days_since_last_order_l5 = mean_days_since_last_order_l5.reset_index() 
mean_days_since_last_order_l5.head()


# In[143]:


mean_days_since_last_order_l5 = mean_days_since_last_order_l5.fillna(0) 
mean_days_since_last_order_l5.head()


# In[144]:


last_five = last_five.merge(mean_days_since_last_order_l5, on=['user_id','product_id'], how='left')
last_five.head()


# In[145]:


del [mean_days_since_last_order_l5]


# In[146]:


last_five = last_five.merge(median_days_since_last_order, on=['user_id','product_id'], how='left')
last_five.head()


# In[147]:


del [median_days_since_last_order]


# In[148]:


avg_pos = op.groupby('product_id').filter(lambda x: x.shape[0]>30)
avg_pos = op.groupby('product_id')['add_to_cart_order'].mean().to_frame('average_position_of_a_product')
avg_pos.head()


# In[149]:


avg_pos = avg_pos.reset_index()
avg_pos.head()


# In[150]:


prd = op.groupby('product_id')['order_id'].count().to_frame('p_total_purchases')
prd.head()


# In[151]:


prd = prd.reset_index()
prd.head()


# In[152]:


prr = op.groupby('product_id').filter(lambda x: x.shape[0] >40)
prr = op.groupby('product_id')['reordered'].mean().to_frame('product_reordered_ratio') #
prr.head()


# In[153]:


prr = prr.reset_index()
prr.head()


# In[154]:


prr = prr.merge(prd, on='product_id', how='right')


# In[155]:


prr = prr.merge(avg_pos, on='product_id', how='left')
prr.head()


# In[156]:


prr = prr.merge(ad, on='product_id', how='left')
prr.head()


# In[157]:


order_size = op.groupby(['user_id', 'order_id'])['product_id'].count().to_frame('size')
order_size = order_size.reset_index()
order_size.head()


# In[158]:


avg_os = order_size.groupby('user_id')['size'].mean().to_frame('average_order_size_for_user')
avg_os = avg_os.reset_index()
avg_os.head()


# In[159]:


last_five = last_five.merge(avg_os, on='user_id', how='left')
last_five.head()


# In[160]:


del [order_size, avg_os]


# In[161]:


aatco = op.groupby(['user_id', 'product_id'])['add_to_cart_order'].mean().to_frame('average_add_to_cart_order')
aatco = aatco.reset_index()
aatco.head()


# In[162]:


adspo = op.groupby('user_id')['days_since_prior_order'].mean().to_frame('average_days_since_prior_order')
adspo = adspo.reset_index()
adspo.head()


# In[163]:


aatco = aatco.merge(adspo, on='user_id', how='left')
aatco.head()


# In[164]:


del [adspo]


# In[165]:


#1.6 num_distinct_product the number of distinct product of user
def f1(x):
    return len(set(x))

num_distinct_prodct = op.groupby('user_id')['product_id'].apply(f1).to_frame('num_distinct_product')
num_distinct_prodct = num_distinct_prodct.reset_index()
num_distinct_prodct.head()




# In[166]:


aatco = aatco.merge(num_distinct_prodct, on='user_id', how='left')
aatco.head()

# In[169]:


del [num_distinct_prodct]

# In[167]:


last_five = last_five.merge(aatco, on=['user_id','product_id'], how='left')
last_five.head()


# In[168]:


del [aatco]


# In[170]:


user = op.groupby('user_id')['order_number'].max().to_frame('u_total_orders')
user = user.reset_index()
user.head()


# In[171]:


urr = op.groupby('user_id')['reordered'].mean().to_frame('user_reordered_ratio') #
urr = urr.reset_index()
urr.head()


# In[172]:


user = user.merge(urr, on='user_id', how='left')

del urr
gc.collect()

user.head() 


# In[219]:


utz = op.groupby(['user_id','timezone'])['order_hour_of_day'].count().to_frame('user_timezone')
utz = utz.reset_index()
utz.head()




user = user.merge(utz, on='user_id', how='left')
utz.head()


# In[ ]:


del [utz]


# In[173]:


uxp = op.groupby(['user_id', 'product_id'])['order_id'].count().to_frame('uxp_total_bought')
uxp.head()


# In[174]:


uxp = uxp.reset_index()
uxp.head()


# In[175]:


times = op.groupby(['user_id', 'product_id'])[['order_id']].count()
times.columns = ['Times_Bought_N']
times.head()


# In[176]:


total_orders = op.groupby('user_id')['order_number'].max().to_frame('total_orders')
total_orders.head()


# In[177]:


first_order_no = op.groupby(['user_id', 'product_id'])['order_number'].min().to_frame('first_order_number')
first_order_no  = first_order_no.reset_index()
first_order_no.head()


# In[178]:


span = pd.merge(total_orders, first_order_no, on='user_id', how='right')
span.head()


# In[179]:


span['Order_Range_D'] = span.total_orders - span.first_order_number + 1
span.head()


# In[180]:


uxp_ratio = pd.merge(times, span, on=['user_id', 'product_id'], how='left')
uxp_ratio.head()


# In[181]:


uxp_ratio['uxp_reorder_ratio'] = uxp_ratio.Times_Bought_N / uxp_ratio.Order_Range_D
uxp_ratio.head()


# In[182]:


uxp_ratio = uxp_ratio.drop(['Times_Bought_N', 'total_orders', 'first_order_number', 'Order_Range_D'], axis=1)
uxp_ratio.head()


# In[183]:


del [times, first_order_no, span]


# In[184]:


uxp = uxp.merge(uxp_ratio, on=['user_id', 'product_id'], how='left')

del uxp_ratio
uxp.head()


# In[185]:


uxp_last5 = op5.groupby(['user_id', 'product_id'])['order_id'].count().to_frame('uxp_total_bought_l5')
uxp_last5.head()


# In[186]:


uxp_last5 = uxp_last5.reset_index()
uxp_last5.head()


# In[187]:


times_l5 = op5.groupby(['user_id', 'product_id'])[['order_id']].count()
times_l5.columns = ['Times_Bought_N_l5']
times_l5.head()


# In[188]:


total_orders_l5 = op5.groupby('user_id')['order_number'].max().to_frame('total_orders_l5')
total_orders_l5.head()


# In[189]:


first_order_no_l5 = op5.groupby(['user_id', 'product_id'])['order_number'].min().to_frame('first_order_number_l5')
first_order_no_l5  = first_order_no_l5.reset_index()
first_order_no_l5.head()


# In[190]:


span_l5 = pd.merge(total_orders_l5, first_order_no_l5, on='user_id', how='right')
span_l5.head()


# In[191]:


span_l5['Order_Range_D_l5'] = span_l5.total_orders_l5 - span_l5.first_order_number_l5 + 1
span_l5.head()


# In[192]:


uxp_ratio_last5 = pd.merge(times_l5, span_l5, on=['user_id', 'product_id'], how='left')
uxp_ratio_last5.head()


# In[193]:


uxp_ratio_last5['uxp_reorder_ratio_last5'] = uxp_ratio_last5.Times_Bought_N_l5 / uxp_ratio_last5.Order_Range_D_l5
uxp_ratio_last5.head()


# In[194]:


uxp_ratio_last5 = uxp_ratio_last5.drop(['Times_Bought_N_l5', 'total_orders_l5', 'first_order_number_l5', 'Order_Range_D_l5'], axis=1)
uxp_ratio_last5.head()


# In[195]:


del [times_l5, first_order_no_l5, span_l5]


# In[196]:


uxp = uxp.merge(uxp_ratio_last5, on=['user_id', 'product_id'], how='left')


# In[197]:


del uxp_ratio_last5
uxp.head()


# In[198]:


o_dow = op.groupby(['user_id','order_dow'])['order_id'].count().to_frame('user_orders_for_each_day')
o_dow = o_dow.reset_index()
o_dow.head()


# In[199]:


o_hod = op.groupby(['user_id','order_hour_of_day'])['order_id'].count().to_frame('user_orders_for_each_hour')
o_hod = o_hod.reset_index()
o_hod.head(25)


# In[200]:


o_dow = pd.merge(o_dow, o_hod, on='user_id', how='outer')
o_dow.head(50)


# In[204]:


uxp = uxp.merge(o_dow, on='user_id', how='left')

del [o_dow, o_hod]
uxp.head()


# In[205]:



uxp = uxp.merge(last_five, on=['user_id', 'product_id'], how='left')

del [last_five]
uxp.head()


# In[ ]:


uxp = uxp.fillna(0)
uxp.head()


# In[ ]:


data = uxp.merge(user, on='user_id', how='left')
data.head()


# In[ ]:


data = data.merge(prr, on='product_id', how='left')
data.head()


# In[ ]:


del op, user, prr, uxp
gc.collect()


# In[ ]:


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


# In[ ]:


data = data.merge(orders_future, on='user_id', how='left')
data.head(10)


# In[ ]:


#Keep only the customers who we know what they bought in their future order
data_train = data[data.eval_set=='train'] #
data_train.head()


# In[ ]:


#Get from order_products_train all the products that the train users bought bought in their future order
data_train = data_train.merge(order_products_train[['product_id','order_id', 'reordered']], on=['product_id','order_id'], how='left' )
data_train.head(15)


# In[ ]:


#Where the previous merge, left a NaN value on reordered column means that the customers they haven't bought the product. We change the value on them to 0.
data_train['reordered'] = data_train['reordered'].fillna(0)
data_train.head(15)


# In[ ]:


#We set user_id and product_id as the index of the DF
data_train = data_train.set_index(['user_id', 'product_id'])
data_train.head(15)


# In[117]:


#We remove all non-predictor variables
data_train = data_train.drop(['eval_set', 'order_id'], axis=1)
data_train.head(15)


# In[ ]:


data_test = data[data.eval_set=='test'] #
data_test.head()


# In[ ]:


#We set user_id and product_id as the index of the DF
data_test = data_test.set_index(['user_id', 'product_id']) #
data_test.head()


# In[ ]:


#We remove all non-predictor variables
data_test = data_test.drop(['eval_set','order_id'], axis=1)
#Check if the data_test DF, has the same number of columns as the data_train DF, excluding the response variable
data_test.head()



# In[ ]:


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

