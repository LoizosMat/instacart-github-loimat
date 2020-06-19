#!/usr/bin/env python
# coding: utf-8

# In[131]:


# For data manipulation
import pandas as pd            
import time
# Garbage Collector to free up memory
import gc                         
gc.enable()                       # Activate 


# In[132]:


orders = pd.read_csv('orders.csv')
order_products_train = pd.read_csv('order_products__train.csv')
order_products_prior = pd.read_csv('order_products__prior.csv')
products = pd.read_csv('products.csv')
aisles = pd.read_csv('aisles.csv')
departments = pd.read_csv('departments.csv')



# In[134]:


# We convert character variables into category. 
# In Python, a categorical variable is called category and has a fixed number of different values
aisles['aisle'] = aisles['aisle'].astype('category')
departments['department'] = departments['department'].astype('category')
orders['eval_set'] = orders['eval_set'].astype('category')
products['product_name'] = products['product_name'].astype('category')


# In[135]:


#Merge the orders DF with order_products_prior by their order_id, keep only these rows with order_id that they are appear on both DFs
op = orders.merge(order_products_prior, on='order_id', how='inner')
op.head()


# In[136]:


opad = op.merge(products, on='product_id', how='left')
opad.head()


# In[137]:


orders_prior=orders[orders.eval_set=='prior']


# In[138]:


aipr = opad.groupby(['aisle_id','product_id'])['order_id'].count().to_frame('aisle_total_orders_of_each_product')
aipr = aipr.reset_index()
aipr.head()


# In[139]:


ais = opad.groupby('aisle_id')['order_id'].count().to_frame('aisle_total_orders')
ais = ais.reset_index()
ais.head()


# In[140]:


aire = opad.groupby('aisle_id')['reordered'].mean().to_frame('aisle_reordered_ratio') #
aire = aire.reset_index()
aire.head()


# In[141]:


aisles = pd.merge(aipr, ais, on='aisle_id', how='inner')
aisles.head()


# In[142]:


avg_ais = opad.groupby('aisle_id')['add_to_cart_order'].mean().to_frame('average_position_of_an_aisle')
avg_ais.head()


# In[143]:


avg_ais = avg_ais.reset_index()
avg_ais.head()


# In[144]:


aisles = aisles.merge(avg_ais, on='aisle_id', how='left')
aisles.head()


# In[145]:


aisles = aisles.merge(aire, on='aisle_id', how='left')
aisles.head()


# In[146]:


ais_avg_time = opad.groupby('aisle_id')['order_hour_of_day'].mean().to_frame('aisles_average_time')
ais_avg_time = ais_avg_time.reset_index()
ais_avg_time.head()


# In[147]:


aisles = aisles.merge(ais_avg_time, on='aisle_id', how='left')
aisles.head()


# In[148]:


depr = opad.groupby(['department_id','product_id'])['order_id'].count().to_frame('department_total_orders_for_each_product')
depr = depr.reset_index()
depr.head()


# In[149]:


dep = opad.groupby('department_id')['order_id'].count().to_frame('department_total_orders')
dep = dep.reset_index()
dep.head()


# In[150]:


dere = opad.groupby('department_id')['reordered'].mean().to_frame('department_reordered_ratio') #
dere = dere.reset_index()
dere.head()


# In[151]:


departments = pd.merge(depr, dep, on='department_id', how='inner')
departments.head()


# In[152]:


avg_dep = opad.groupby('department_id')['add_to_cart_order'].mean().to_frame('average_position_of_a_department')
avg_dep = avg_dep.reset_index()
avg_dep.head()


# In[153]:


departments = departments.merge(avg_dep, on='department_id', how='left')
departments.head()


# In[154]:


departments = departments.merge(dere, on='department_id', how='left')
departments.head()


# In[155]:


dep_avg_time = opad.groupby('department_id')['order_hour_of_day'].mean().to_frame('department_average_time')
dep_avg_time = dep_avg_time.reset_index()
dep_avg_time.head(40)


# In[156]:


departments = departments.merge(dep_avg_time, on='department_id', how='left')
departments.head()


# In[157]:


ad = pd.merge(aisles, departments, on='product_id', how='inner')
ad.head()


# In[158]:


del [aipr, depr, aisles, departments]


# In[159]:


uxp = op.groupby(['user_id', 'product_id'])['order_id'].count().to_frame('uxp_total_bought')
uxp.head()


# In[160]:


uxp = uxp.reset_index()
uxp.head()


# In[161]:


times = op.groupby(['user_id', 'product_id'])[['order_id']].count()
times.columns = ['Times_Bought_N']
times.head()


# In[162]:


total_orders = op.groupby('user_id')['order_number'].max().to_frame('total_orders')
total_orders.head()


# In[163]:


first_order_no = op.groupby(['user_id', 'product_id'])['order_number'].min().to_frame('first_order_number')
first_order_no  = first_order_no.reset_index()
first_order_no.head()


# In[164]:


span = pd.merge(total_orders, first_order_no, on='user_id', how='right')
span.head()


# In[165]:


span['Order_Range_D'] = span.total_orders - span.first_order_number + 1
span.head()


# In[166]:


uxp_ratio = pd.merge(times, span, on=['user_id', 'product_id'], how='left')
uxp_ratio.head()


# In[167]:


uxp_ratio['uxp_reorder_ratio'] = uxp_ratio.Times_Bought_N / uxp_ratio.Order_Range_D
uxp_ratio.head()


# In[168]:


uxp_ratio = uxp_ratio.drop(['Times_Bought_N', 'total_orders', 'first_order_number', 'Order_Range_D'], axis=1)
uxp_ratio.head()


# In[169]:


del [times, first_order_no, span]


# In[170]:


uxp = uxp.merge(uxp_ratio, on=['user_id', 'product_id'], how='left')

del uxp_ratio
uxp.head()


# In[171]:


op['order_number_back'] = op.groupby('user_id')['order_number'].transform(max) - op.order_number +1
op5 = op[op.order_number_back <= 5]
op5.head()
last_five = op5.groupby(['user_id','product_id'])[['order_id']].count()
last_five.columns = ['times_last5']
last_five = last_five.reset_index()
last_five.head(10)


# In[172]:


times_l5 = op5.groupby(['user_id', 'product_id'])[['order_id']].count()
times_l5.columns = ['Times_Bought_N_l5']
times_l5.head()


# In[173]:


total_orders_l5 = op5.groupby('user_id')['order_number'].max().to_frame('total_orders_l5')
total_orders_l5.head()


# In[174]:


first_order_no_l5 = op5.groupby(['user_id', 'product_id'])['order_number'].min().to_frame('first_order_number_l5')
first_order_no_l5  = first_order_no_l5.reset_index()
first_order_no_l5.head()


# In[175]:


span_l5 = pd.merge(total_orders_l5, first_order_no_l5, on='user_id', how='right')
span_l5.head()


# In[176]:


span_l5['Order_Range_D_l5'] = span_l5.total_orders_l5 - span_l5.first_order_number_l5 + 1
span_l5.head()


# In[177]:


uxp_ratio_last5 = pd.merge(times_l5, span_l5, on=['user_id', 'product_id'], how='left')
uxp_ratio_last5.head()


# In[178]:


uxp_ratio_last5['uxp_reorder_ratio_last5'] = uxp_ratio_last5.Times_Bought_N_l5 / uxp_ratio_last5.Order_Range_D_l5
uxp_ratio_last5.head()


# In[179]:


uxp_ratio_last5 = uxp_ratio_last5.drop(['Times_Bought_N_l5', 'total_orders_l5', 'first_order_number_l5', 'Order_Range_D_l5'], axis=1)
uxp_ratio_last5.head()


# In[180]:


del [times_l5, first_order_no_l5, span_l5]


# In[181]:


last_five = last_five.merge(uxp_ratio_last5, on=['user_id', 'product_id'], how='left')


# In[182]:


del uxp_ratio_last5
last_five.head()


# In[183]:


last_five['times_last5_ratio'] = last_five.times_last5 / 5
last_five.head()


# In[184]:


max_days_since_last_order_l5 = op5.groupby(['user_id','product_id'])[['days_since_prior_order']].max()
max_days_since_last_order_l5.columns = ['max_days_since_last5'] 
max_days_since_last_order_l5 = max_days_since_last_order_l5.reset_index() 
max_days_since_last_order_l5.head()


# In[185]:


max_days_since_last_order_l5 = max_days_since_last_order_l5.fillna(0)
max_days_since_last_order_l5.head()


# In[186]:


last_five = last_five.merge(max_days_since_last_order_l5, on=['user_id','product_id'], how='left')
last_five.head()


# In[187]:


del [max_days_since_last_order_l5]


# In[188]:


max_days_since_last_order = op.groupby(['user_id','product_id'])[['days_since_prior_order']].max()
max_days_since_last_order.columns = ['max_days_since_last'] 
max_days_since_last_order = max_days_since_last_order.reset_index() 
max_days_since_last_order.head()


# In[189]:


max_days_since_last_order = max_days_since_last_order.fillna(0)
max_days_since_last_order.head()


# In[190]:


days_since_last_order = op.groupby(['user_id','product_id'])[['days_since_prior_order']].count()
days_since_last_order.columns = ['days_since_last_order'] 
days_since_last_order = days_since_last_order.reset_index() 
days_since_last_order.head()


# In[191]:


days_since_last_order = days_since_last_order.fillna(0)
days_since_last_order.head()


# In[192]:


days_last_order_max = pd.merge(days_since_last_order, max_days_since_last_order , on=['user_id', 'product_id'], how='left')
days_last_order_max.head()


# In[193]:


days_last_order_max['days_last_order_max'] = days_last_order_max.days_since_last_order / days_last_order_max.max_days_since_last
days_last_order_max.head()


# In[194]:


del [days_since_last_order, max_days_since_last_order]


# In[195]:


uxp = uxp.merge(days_last_order_max, on=['user_id','product_id'], how='left')
uxp.head()


# In[196]:


median_days_since_last_order_l5 = op5.groupby(['user_id','product_id'])[['days_since_prior_order']].median()
median_days_since_last_order_l5.columns = ['median_days_since_last5'] 
median_days_since_last_order_l5 = median_days_since_last_order_l5.reset_index() 
median_days_since_last_order_l5.head()


# In[197]:


median_days_since_last_order_l5 = median_days_since_last_order_l5.fillna(0)
median_days_since_last_order_l5.head()


# In[198]:


median_days_since_last_order = op.groupby(['user_id','product_id'])[['days_since_prior_order']].median()
median_days_since_last_order.columns = ['median_days_since_last'] 
median_days_since_last_order = median_days_since_last_order.reset_index() 
median_days_since_last_order.head()


# In[199]:


median_days_since_last_order = median_days_since_last_order.fillna(0)
median_days_since_last_order.head()


# In[200]:


uxp = uxp.merge(median_days_since_last_order, on=['user_id','product_id'], how='left')
uxp.head()


# In[201]:


last_five = last_five.merge(median_days_since_last_order_l5, on=['user_id','product_id'], how='left')
last_five.head()


# In[202]:


del [median_days_since_last_order_l5]


# In[203]:


min_days_since_last_order_l5 = op5.groupby(['user_id','product_id'])[['days_since_prior_order']].min()
min_days_since_last_order_l5.columns = ['min_days_since_last5'] 
min_days_since_last_order_l5 = min_days_since_last_order_l5.reset_index() 
min_days_since_last_order_l5.head()


# In[204]:


min_days_since_last_order_l5 = min_days_since_last_order_l5.fillna(0)
min_days_since_last_order_l5.head()


# In[205]:


last_five = last_five.merge(min_days_since_last_order_l5, on=['user_id','product_id'], how='left')
last_five.head()


# In[206]:


del [min_days_since_last_order_l5]


# In[207]:


mean_days_since_last_order_l5 = op5.groupby(['user_id','product_id'])[['days_since_prior_order']].mean()
mean_days_since_last_order_l5.columns = ['mean_days_since_last5'] 
mean_days_since_last_order_l5 = mean_days_since_last_order_l5.reset_index() 
mean_days_since_last_order_l5.head()


# In[208]:


mean_days_since_last_order_l5 = mean_days_since_last_order_l5.fillna(0) 
mean_days_since_last_order_l5.head()


# In[209]:


last_five = last_five.merge(mean_days_since_last_order_l5, on=['user_id','product_id'], how='left')
last_five.head()


# In[210]:


del [mean_days_since_last_order_l5]


# In[211]:


del [median_days_since_last_order]


# In[212]:


aatco_l5 = op5.groupby(['user_id', 'product_id'])['add_to_cart_order'].mean().to_frame('average_add_to_cart_last5')
aatco_l5 = aatco_l5.reset_index()
aatco_l5.head()


# In[213]:


aatco_l5 = aatco_l5.fillna(0) 
aatco_l5.head()


# In[214]:


last_five = last_five.merge(aatco_l5, on=['user_id','product_id'], how='left')
last_five.head()


# In[215]:


del [aatco_l5]


# In[216]:


uxp_avg_time_l5 = op5.groupby(['user_id', 'product_id'])['order_hour_of_day'].mean().to_frame('uxp_average_time_last5')
uxp_avg_time_l5 = uxp_avg_time_l5.reset_index()
uxp_avg_time_l5.head()


# In[217]:


last_five = last_five.merge(uxp_avg_time_l5, on=['user_id','product_id'], how='left')
last_five.head()


# In[218]:


del [uxp_avg_time_l5]


# In[219]:


avg_pos = op.groupby('product_id').filter(lambda x: x.shape[0]>30)
avg_pos = op.groupby('product_id')['add_to_cart_order'].mean().to_frame('average_position_of_a_product')
avg_pos.head()


# In[220]:


avg_pos = avg_pos.reset_index()
avg_pos.head()


# In[221]:


prd = op.groupby('product_id')['order_id'].count().to_frame('p_total_purchases')
prd.head()


# In[222]:


prd = prd.reset_index()
prd.head()


# In[223]:


def f1(x):
    return len(set(x))


# In[224]:


product_of_user_num = op.groupby('product_id')['user_id'].apply(f1).to_frame('product_of_user_num')
product_of_user_num = product_of_user_num.reset_index()
product_of_user_num.head()


# In[225]:


prd = prd.merge(product_of_user_num, on='product_id', how='left')
prd.head()


# In[226]:


prr = op.groupby('product_id').filter(lambda x: x.shape[0] >40)
prr = op.groupby('product_id')['reordered'].mean().to_frame('product_reordered_ratio') #
prr.head()


# In[227]:


prr = prr.reset_index()
prr.head()


# In[228]:


prr = prr.merge(prd, on='product_id', how='right')


# In[229]:


prr = prr.merge(avg_pos, on='product_id', how='left')
prr.head()


# In[230]:


pr_avg_time = opad.groupby('product_id')['order_hour_of_day'].mean().to_frame('products_average_time')
pr_avg_time = pr_avg_time.reset_index()
pr_avg_time.head(40)


# In[231]:


prr = prr.merge(pr_avg_time, on='product_id', how='left')
prr.head()


# In[232]:


product_of_second_order_num = op[op.reordered==1].groupby('product_id')['order_id'].count().to_frame('product_of_second_order_num')
product_of_second_order_num = product_of_second_order_num.reset_index()
product_of_second_order_num.head()


# In[233]:


prr = prr.merge(product_of_second_order_num, on='product_id', how='left')
prr.head()


# In[234]:


product_of_second_user_num = op[op.reordered==1].groupby('product_id')['user_id'].apply(f1).to_frame('product_of_second_user_num')
product_of_second_user_num = product_of_second_user_num.reset_index()
product_of_second_user_num.head()


# In[235]:


prr = prr.merge(product_of_second_user_num, on='product_id', how='left')
prr.head()


# In[236]:


prr = prr.merge(ad, on='product_id', how='left')
prr.head()


# In[237]:


aatco = op.groupby(['user_id', 'product_id'])['add_to_cart_order'].mean().to_frame('average_add_to_cart_order')
aatco = aatco.reset_index()
aatco.head()


# In[238]:


uxp = uxp.merge(aatco, on=['user_id', 'product_id'], how='left')
uxp.head()


# In[239]:


uxp_avg_time = op.groupby(['user_id', 'product_id'])['order_hour_of_day'].mean().to_frame('uxp_average_time')
uxp_avg_time = uxp_avg_time.reset_index()
uxp_avg_time.head()


# In[240]:


uxp = uxp.merge(uxp_avg_time, on=['user_id', 'product_id'], how='left')
uxp.head()


# In[241]:


del [aatco, uxp_avg_time]


# In[242]:


user = op.groupby('user_id')['order_number'].max().to_frame('u_total_orders')
user = user.reset_index()
user.head()


# In[243]:


order_size = op.groupby('order_id')['add_to_cart_order'].max().to_frame('order_size')
order_size = order_size.reset_index()
order_size.head()


# In[244]:


avg_size = pd.merge(order_size, orders, on='order_id', how='left')
avg_size.head()


# In[245]:


avg_os = avg_size.groupby('user_id')['order_size'].mean().to_frame('average_order_size_for_user')
avg_os = avg_os.reset_index()
avg_os.head()


# In[246]:


user = user.merge(avg_os, on='user_id', how='left')
user.head()


# In[247]:


max_os = avg_size.groupby('user_id')['order_size'].max().to_frame('max_order_size_for_user')
max_os = max_os.reset_index()
max_os.head()


# In[248]:


user = user.merge(max_os, on='user_id', how='left')
user.head()


# In[249]:


del [order_size, avg_os, avg_size, max_os]


# In[250]:


adspo = op.groupby('user_id')['days_since_prior_order'].mean().to_frame('average_days_since_prior_order')
adspo = adspo.reset_index()
adspo.head()


# In[251]:


user = user.merge(adspo, on='user_id', how='left')
user.head()


# In[252]:


del [adspo]


# In[253]:


#1.6 num_distinct_product the number of distinct product of user
num_distinct_prodct = op.groupby('user_id')['product_id'].apply(f1).to_frame('num_distinct_product')
num_distinct_prodct = num_distinct_prodct.reset_index()
num_distinct_prodct.head()


# In[254]:


user = user.merge(num_distinct_prodct, on='user_id', how='left')
user.head()


# In[255]:


data2 = op.groupby('user_id')['order_number'].max().to_frame('data2')
data2 = data2.reset_index()
data2.head()


# In[256]:


lor = pd.merge(op, user, on='user_id',how='left')
lor = lor[lor.order_number==lor.u_total_orders]
lor.head()


# In[257]:


lorr = lor.groupby('user_id')['reordered'].mean().to_frame('lor')
lorr = lorr.reset_index()
lorr.head()


# In[258]:


urr = op.groupby('user_id')['reordered'].mean().to_frame('user_reordered_ratio') #
urr = urr.reset_index()
urr.head()


# In[259]:


user = user.merge(urr, on='user_id', how='left')

del urr
gc.collect()

user.head() 


# In[260]:


gmcs = op.groupby(['user_id'])['order_id'].count().to_frame('gmcs')
gmcs = gmcs.reset_index()
gmcs.head()


# In[261]:


user = user.merge(gmcs, on='user_id', how='left')
user.head()


# In[262]:


user = user.merge(lorr, on='user_id', how='left')
user.head()


# In[263]:


user_avg_time = op.groupby('user_id')['order_hour_of_day'].mean().to_frame('user_average_time')
user_avg_time = user_avg_time.reset_index()
user_avg_time.head()


# In[264]:


user = user.merge(user_avg_time, on='user_id', how='left')
user.head()


# In[265]:


ex = op.groupby('user_id')['add_to_cart_order'].mean().to_frame('ex')
ex = ex.reset_index()
ex.head()


# In[266]:


user = user.merge(ex, on='user_id', how='left')
user.head()


# In[267]:


del [user_avg_time, ex, lorr, gmcs]


# In[268]:


data3 = pd.merge(opad, data2, on='user_id',how='left')
data3.head()


# In[269]:


uxp = uxp.merge(last_five, on=['user_id', 'product_id'], how='left')

del [last_five]
uxp.head()


# In[270]:


uxp = uxp.fillna(0)
uxp.head()


# In[271]:


data = uxp.merge(user, on='user_id', how='left')
data.head()


# In[272]:


data = data.merge(prr, on='product_id', how='left')
data.head()


# In[273]:


del op, op5, user, prr, uxp, ad
gc.collect()


# In[274]:


## First approach:
# In two steps keep only the future orders from all customers: train & test 
orders_future = orders[((orders.eval_set=='train') | (orders.eval_set=='test'))]
orders_future = orders_future[ ['user_id', 'eval_set', 'order_id'] ]
orders_future.head(10)


# In[275]:


data = data.merge(orders_future, on='user_id', how='left')
data.head(10)


# In[276]:


#Keep only the customers who we know what they bought in their future order
data_train = data[data.eval_set=='train'] #
data_train.head()


# In[277]:


#Get from order_products_train all the products that the train users bought bought in their future order
data_train = data_train.merge(order_products_train[['product_id','order_id', 'reordered']], on=['product_id','order_id'], how='left' )
data_train.head(15)


# In[278]:


#Where the previous merge, left a NaN value on reordered column means that the customers they haven't bought the product. We change the value on them to 0.
data_train['reordered'] = data_train['reordered'].fillna(0)
data_train.head(15)


# In[279]:


#We set user_id and product_id as the index of the DF
data_train = data_train.set_index(['user_id', 'product_id'])
data_train.head(15)


# In[280]:


#We remove all non-predictor variables
data_train = data_train.drop(['eval_set', 'order_id', 'aisle_id', 'department_id'], axis=1)
data_train.head(15)


# In[281]:


data_test = data[data.eval_set=='test'] #
data_test.head()


# In[282]:


#We set user_id and product_id as the index of the DF
data_test = data_test.set_index(['user_id', 'product_id']) #
data_test.head()


# In[283]:


#We remove all non-predictor variables
data_test = data_test.drop(['eval_set','order_id','aisle_id','department_id'], axis=1)
#Check if the data_test DF, has the same number of columns as the data_train DF, excluding the response variable
data_test.head()


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


##########################################
## SPLIT DF TO: X_train, y_train (axis=1)
##########################################
X_train, y_train = data_train.drop('reordered', axis=1), data_train.reordered

paramGrid = {'n_estimators':[800], 
              'max_depth':[6],
             'learning_rate':[0.01,0.02,0.03],
             'min_child_weight':[1],
             'colsample_bytree':[1],
             'subsample':[1],
             }

##############
## INSTANTIATE XGBClassifier()
########################################
xgbc = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', gpu_id=0, tree_method= 'gpu_hist')

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

# Store the model for prediction (chapter 5).
model = gridsearch.best_estimator_

# Delete X_train , y_train
del [X_train, y_train]


# In[117]:


model.get_params()




# In[ ]:


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

