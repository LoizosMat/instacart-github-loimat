#!/usr/bin/env python
# coding: utf-8

# In[16]:


# For data manipulation
import pandas as pd            
import time
import numpy as np
# Garbage Collector to free up memory
import gc                         
gc.enable()                       # Activate 


# In[17]:


orders = pd.read_csv('orders.csv')
order_products_train = pd.read_csv('order_products__train.csv')
order_products_prior = pd.read_csv('order_products__prior.csv')
products = pd.read_csv('products.csv')
aisles = pd.read_csv('aisles.csv')
departments = pd.read_csv('departments.csv')



# In[19]:


# We convert character variables into category. 
# In Python, a categorical variable is called category and has a fixed number of different values
aisles['aisle'] = aisles['aisle'].astype('category')
departments['department'] = departments['department'].astype('category')
orders['eval_set'] = orders['eval_set'].astype('category')
products['product_name'] = products['product_name'].astype('category')


# In[20]:


#Merge the orders DF with order_products_prior by their order_id, keep only these rows with order_id that they are appear on both DFs
op = orders.merge(order_products_prior, on='order_id', how='inner')
op.head()


# In[21]:


opad = op.merge(products, on='product_id', how='left')
opad.head()


# In[7]:


orders_prior=orders[orders.eval_set=='prior']


# In[8]:


aipr = opad.groupby(['aisle_id','product_id'])['order_id'].count().to_frame('aisle_total_orders_of_each_product')
aipr = aipr.reset_index()
aipr.head()


# In[9]:


ais = opad.groupby('aisle_id')['order_id'].count().to_frame('aisle_total_orders')
ais = ais.reset_index()
ais.head()


# In[10]:


aire = opad.groupby('aisle_id')['reordered'].mean().to_frame('aisle_reordered_ratio') #
aire = aire.reset_index()
aire.head()


# In[11]:


aisles = pd.merge(aipr, ais, on='aisle_id', how='inner')
aisles.head()


# In[12]:


avg_ais = opad.groupby('aisle_id')['add_to_cart_order'].mean().to_frame('average_position_of_an_aisle')
avg_ais.head()


# In[13]:


avg_ais = avg_ais.reset_index()
avg_ais.head()


# In[14]:


aisles = aisles.merge(avg_ais, on='aisle_id', how='left')
aisles.head()


# In[15]:


aisles = aisles.merge(aire, on='aisle_id', how='left')
aisles.head()


# In[16]:


ais_avg_time = opad.groupby('aisle_id')['order_hour_of_day'].mean().to_frame('aisles_average_time')
ais_avg_time = ais_avg_time.reset_index()
ais_avg_time.head()


# In[17]:


aisles = aisles.merge(ais_avg_time, on='aisle_id', how='left')
aisles.head()


# In[18]:


depr = opad.groupby(['department_id','product_id'])['order_id'].count().to_frame('department_total_orders_for_each_product')
depr = depr.reset_index()
depr.head()


# In[19]:


dep = opad.groupby('department_id')['order_id'].count().to_frame('department_total_orders')
dep = dep.reset_index()
dep.head()


# In[20]:


dere = opad.groupby('department_id')['reordered'].mean().to_frame('department_reordered_ratio') #
dere = dere.reset_index()
dere.head()


# In[21]:


departments = pd.merge(depr, dep, on='department_id', how='inner')
departments.head()


# In[22]:


avg_dep = opad.groupby('department_id')['add_to_cart_order'].mean().to_frame('average_position_of_a_department')
avg_dep = avg_dep.reset_index()
avg_dep.head()


# In[23]:


departments = departments.merge(avg_dep, on='department_id', how='left')
departments.head()


# In[24]:


departments = departments.merge(dere, on='department_id', how='left')
departments.head()


# In[25]:


dep_avg_time = opad.groupby('department_id')['order_hour_of_day'].mean().to_frame('department_average_time')
dep_avg_time = dep_avg_time.reset_index()
dep_avg_time.head(40)


# In[26]:


departments = departments.merge(dep_avg_time, on='department_id', how='left')
departments.head()


# In[27]:


ad = pd.merge(aisles, departments, on='product_id', how='inner')
ad.head()


# In[28]:


del [aipr, depr, aisles, departments]


# In[29]:


uxp = op.groupby(['user_id', 'product_id'])['order_id'].count().to_frame('uxp_total_bought')
uxp.head()


# In[30]:


uxp = uxp.reset_index()
uxp.head()


# In[31]:


times = op.groupby(['user_id', 'product_id'])[['order_id']].count()
times.columns = ['Times_Bought_N']
times.head()


# In[32]:


total_orders = op.groupby('user_id')['order_number'].max().to_frame('total_orders')
total_orders.head()


# In[33]:


first_order_no = op.groupby(['user_id', 'product_id'])['order_number'].min().to_frame('first_order_number')
first_order_no  = first_order_no.reset_index()
first_order_no.head()


# In[34]:


span = pd.merge(total_orders, first_order_no, on='user_id', how='right')
span.head()


# In[35]:


span['Order_Range_D'] = span.total_orders - span.first_order_number + 1
span.head()


# In[36]:


uxp_ratio = pd.merge(times, span, on=['user_id', 'product_id'], how='left')
uxp_ratio.head()


# In[37]:


uxp_ratio['uxp_reorder_ratio'] = uxp_ratio.Times_Bought_N / uxp_ratio.Order_Range_D
uxp_ratio.head()


# In[38]:


uxp_ratio = uxp_ratio.drop(['Times_Bought_N', 'total_orders', 'first_order_number', 'Order_Range_D'], axis=1)
uxp_ratio.head()


# In[39]:


del [times, first_order_no, span]


# In[40]:


uxp = uxp.merge(uxp_ratio, on=['user_id', 'product_id'], how='left')

del uxp_ratio
uxp.head()


# In[41]:


op['order_number_back'] = op.groupby('user_id')['order_number'].transform(max) - op.order_number +1
op5 = op[op.order_number_back <= 5]
op5.head()
last_five = op5.groupby(['user_id','product_id'])[['order_id']].count()
last_five.columns = ['times_last5']
last_five = last_five.reset_index()
last_five.head(10)


# In[42]:


times_l5 = op5.groupby(['user_id', 'product_id'])[['order_id']].count()
times_l5.columns = ['Times_Bought_N_l5']
times_l5.head()


# In[43]:


total_orders_l5 = op5.groupby('user_id')['order_number'].max().to_frame('total_orders_l5')
total_orders_l5.head()


# In[44]:


first_order_no_l5 = op5.groupby(['user_id', 'product_id'])['order_number'].min().to_frame('first_order_number_l5')
first_order_no_l5  = first_order_no_l5.reset_index()
first_order_no_l5.head()


# In[45]:


span_l5 = pd.merge(total_orders_l5, first_order_no_l5, on='user_id', how='right')
span_l5.head()


# In[46]:


span_l5['Order_Range_D_l5'] = span_l5.total_orders_l5 - span_l5.first_order_number_l5 + 1
span_l5.head()


# In[47]:


uxp_ratio_last5 = pd.merge(times_l5, span_l5, on=['user_id', 'product_id'], how='left')
uxp_ratio_last5.head()


# In[48]:


uxp_ratio_last5['uxp_reorder_ratio_last5'] = uxp_ratio_last5.Times_Bought_N_l5 / uxp_ratio_last5.Order_Range_D_l5
uxp_ratio_last5.head()


# In[49]:


uxp_ratio_last5 = uxp_ratio_last5.drop(['Times_Bought_N_l5', 'total_orders_l5', 'first_order_number_l5', 'Order_Range_D_l5'], axis=1)
uxp_ratio_last5.head()


# In[50]:


del [times_l5, first_order_no_l5, span_l5]


# In[51]:


last_five = last_five.merge(uxp_ratio_last5, on=['user_id', 'product_id'], how='left')


# In[52]:


del uxp_ratio_last5
last_five.head()


# In[53]:


last_five['times_last5_ratio'] = last_five.times_last5 / 5
last_five.head()


# In[54]:


max_days_since_last_order_l5 = op5.groupby(['user_id','product_id'])[['days_since_prior_order']].max()
max_days_since_last_order_l5.columns = ['max_days_since_last5'] 
max_days_since_last_order_l5 = max_days_since_last_order_l5.reset_index() 
max_days_since_last_order_l5.head()


# In[55]:


max_days_since_last_order_l5 = max_days_since_last_order_l5.fillna(0)
max_days_since_last_order_l5.head()


# In[56]:


last_five = last_five.merge(max_days_since_last_order_l5, on=['user_id','product_id'], how='left')
last_five.head()


# In[57]:


del [max_days_since_last_order_l5]


# In[58]:


max_days_since_last_order = op.groupby(['user_id','product_id'])[['days_since_prior_order']].max()
max_days_since_last_order.columns = ['max_days_since_last'] 
max_days_since_last_order = max_days_since_last_order.reset_index() 
max_days_since_last_order.head()


# In[59]:


max_days_since_last_order = max_days_since_last_order.fillna(0)
max_days_since_last_order.head()


# In[60]:


days_since_last_order = op.groupby(['user_id','product_id'])[['days_since_prior_order']].count()
days_since_last_order.columns = ['days_since_last_order'] 
days_since_last_order = days_since_last_order.reset_index() 
days_since_last_order.head()


# In[61]:


days_since_last_order = days_since_last_order.fillna(0)
days_since_last_order.head()


# In[62]:


days_last_order_max = pd.merge(days_since_last_order, max_days_since_last_order , on=['user_id', 'product_id'], how='left')
days_last_order_max.head()


# In[63]:


days_last_order_max['days_last_order_max'] = days_last_order_max.days_since_last_order / days_last_order_max.max_days_since_last
days_last_order_max.head()


# In[64]:


del [days_since_last_order, max_days_since_last_order]


# In[65]:


uxp = uxp.merge(days_last_order_max, on=['user_id','product_id'], how='left')
uxp.head()


# In[66]:


median_days_since_last_order_l5 = op5.groupby(['user_id','product_id'])[['days_since_prior_order']].median()
median_days_since_last_order_l5.columns = ['median_days_since_last5'] 
median_days_since_last_order_l5 = median_days_since_last_order_l5.reset_index() 
median_days_since_last_order_l5.head()


# In[67]:


median_days_since_last_order_l5 = median_days_since_last_order_l5.fillna(0)
median_days_since_last_order_l5.head()


# In[68]:


median_days_since_last_order = op.groupby(['user_id','product_id'])[['days_since_prior_order']].median()
median_days_since_last_order.columns = ['median_days_since_last'] 
median_days_since_last_order = median_days_since_last_order.reset_index() 
median_days_since_last_order.head()


# In[69]:


median_days_since_last_order = median_days_since_last_order.fillna(0)
median_days_since_last_order.head()


# In[70]:


uxp = uxp.merge(median_days_since_last_order, on=['user_id','product_id'], how='left')
uxp.head()


# In[71]:


last_five = last_five.merge(median_days_since_last_order_l5, on=['user_id','product_id'], how='left')
last_five.head()


# In[72]:


del [median_days_since_last_order_l5]


# In[73]:


min_days_since_last_order_l5 = op5.groupby(['user_id','product_id'])[['days_since_prior_order']].min()
min_days_since_last_order_l5.columns = ['min_days_since_last5'] 
min_days_since_last_order_l5 = min_days_since_last_order_l5.reset_index() 
min_days_since_last_order_l5.head()


# In[74]:


min_days_since_last_order_l5 = min_days_since_last_order_l5.fillna(0)
min_days_since_last_order_l5.head()


# In[75]:


last_five = last_five.merge(min_days_since_last_order_l5, on=['user_id','product_id'], how='left')
last_five.head()


# In[76]:


del [min_days_since_last_order_l5]


# In[77]:


mean_days_since_last_order_l5 = op5.groupby(['user_id','product_id'])[['days_since_prior_order']].mean()
mean_days_since_last_order_l5.columns = ['mean_days_since_last5'] 
mean_days_since_last_order_l5 = mean_days_since_last_order_l5.reset_index() 
mean_days_since_last_order_l5.head()


# In[78]:


mean_days_since_last_order_l5 = mean_days_since_last_order_l5.fillna(0) 
mean_days_since_last_order_l5.head()


# In[79]:


last_five = last_five.merge(mean_days_since_last_order_l5, on=['user_id','product_id'], how='left')
last_five.head()


# In[80]:


del [mean_days_since_last_order_l5]


# In[81]:


del [median_days_since_last_order]


# In[82]:


aatco_l5 = op5.groupby(['user_id', 'product_id'])['add_to_cart_order'].mean().to_frame('average_add_to_cart_last5')
aatco_l5 = aatco_l5.reset_index()
aatco_l5.head()


# In[83]:


aatco_l5 = aatco_l5.fillna(0) 
aatco_l5.head()


# In[84]:


last_five = last_five.merge(aatco_l5, on=['user_id','product_id'], how='left')
last_five.head()


# In[85]:


del [aatco_l5]


# In[86]:


uxp_avg_time_l5 = op5.groupby(['user_id', 'product_id'])['order_hour_of_day'].mean().to_frame('uxp_average_time_last5')
uxp_avg_time_l5 = uxp_avg_time_l5.reset_index()
uxp_avg_time_l5.head()


# In[87]:


last_five = last_five.merge(uxp_avg_time_l5, on=['user_id','product_id'], how='left')
last_five.head()


# In[88]:


urr_l5 = op5.groupby('user_id')['reordered'].mean().to_frame('user_reorder_ratio_last5')
urr_l5 = urr_l5.reset_index()
urr_l5.head()


# In[89]:


last_five = last_five.merge(urr_l5, on='user_id', how='left')
last_five.head()


# In[90]:


user_avg_time_l5 = op5.groupby('user_id')['order_hour_of_day'].mean().to_frame('user_average_time_last5')
user_avg_time_l5 = user_avg_time_l5.reset_index()
user_avg_time_l5.head()


# In[91]:


last_five = last_five.merge(user_avg_time_l5, on='user_id', how='left')
last_five.head()


# In[92]:


user_avg_atco_l5 = op5.groupby('user_id')['add_to_cart_order'].mean().to_frame('user_average_add_to_cart_order_last5')
user_avg_atco_l5 = user_avg_atco_l5.reset_index()
user_avg_atco_l5.head()


# In[93]:


last_five = last_five.merge(user_avg_atco_l5, on='user_id', how='left')
last_five.head()


# In[94]:


adspo_l5 = op5.groupby('user_id')['days_since_prior_order'].mean().to_frame('average_days_since_prior_order_l5')
adspo_l5 = adspo_l5.reset_index()
adspo_l5.head()


# In[95]:


last_five = last_five.merge(adspo_l5, on='user_id', how='left')
last_five.head()


# In[96]:


avg_pos = op.groupby('product_id').filter(lambda x: x.shape[0]>30)
avg_pos = op.groupby('product_id')['add_to_cart_order'].mean().to_frame('average_position_of_a_product')
avg_pos.head()


# In[97]:


avg_pos = avg_pos.reset_index()
avg_pos.head()


# In[98]:


prd = op.groupby('product_id')['order_id'].count().to_frame('p_total_purchases')
prd.head()


# In[99]:


prd = prd.reset_index()
prd.head()


# In[100]:


def f1(x):
    return len(set(x))


# In[104]:


uup = op.groupby('product_id')['user_id'].apply(f1).to_frame('unique_users_of_a_product')
uup = uup.reset_index()
uup.head()


# In[105]:


prd = prd.merge(uup, on='product_id', how='left')
prd.head()


# In[106]:


prr = op.groupby('product_id').filter(lambda x: x.shape[0] >40)
prr = op.groupby('product_id')['reordered'].mean().to_frame('product_reordered_ratio') #
prr.head()


# In[107]:


prr = prr.reset_index()
prr.head()


# In[108]:


prr = prr.merge(prd, on='product_id', how='right')


# In[109]:


prr = prr.merge(avg_pos, on='product_id', how='left')
prr.head()


# In[110]:


pr_avg_time = opad.groupby('product_id')['order_hour_of_day'].mean().to_frame('products_average_time')
pr_avg_time = pr_avg_time.reset_index()
pr_avg_time.head(40)


# In[111]:


prr = prr.merge(pr_avg_time, on='product_id', how='left')
prr.head()


# In[112]:


pr_second = op[op.reordered==1].groupby('product_id')['order_id'].count().to_frame('products_reordered_second_time')
pr_second = pr_second.reset_index()
pr_second.head()


# In[113]:


prr = prr.merge(pr_second, on='product_id', how='left')
prr.head()


# In[114]:


uurp = op[op.reordered==1].groupby('product_id')['user_id'].apply(f1).to_frame('unique_users_of_reordered_products')
uurp = uurp.reset_index()
uurp.head()


# In[115]:


prr = prr.merge(uurp, on='product_id', how='left')
prr.head()


# In[116]:


prr = prr.merge(ad, on='product_id', how='left')
prr.head()


# In[117]:


aatco = op.groupby(['user_id', 'product_id'])['add_to_cart_order'].mean().to_frame('average_add_to_cart_order')
aatco = aatco.reset_index()
aatco.head()


# In[118]:


uxp = uxp.merge(aatco, on=['user_id', 'product_id'], how='left')
uxp.head()


# In[119]:


uxp_avg_time = op.groupby(['user_id', 'product_id'])['order_hour_of_day'].mean().to_frame('uxp_average_time')
uxp_avg_time = uxp_avg_time.reset_index()
uxp_avg_time.head()


# In[120]:


uxp = uxp.merge(uxp_avg_time, on=['user_id', 'product_id'], how='left')
uxp.head()


# In[121]:


del [aatco, uxp_avg_time]


# In[122]:


user = op.groupby('user_id')['order_number'].max().to_frame('u_total_orders')
user = user.reset_index()
user.head()


# In[123]:


order_size = op.groupby('order_id')['add_to_cart_order'].max().to_frame('order_size')
order_size = order_size.reset_index()
order_size.head()


# In[124]:


avg_size = pd.merge(order_size, orders, on='order_id', how='left')
avg_size.head()


# In[125]:


avg_os = avg_size.groupby('user_id')['order_size'].mean().to_frame('average_order_size_for_user')
avg_os = avg_os.reset_index()
avg_os.head()


# In[126]:


user = user.merge(avg_os, on='user_id', how='left')
user.head()


# In[127]:


max_os = avg_size.groupby('user_id')['order_size'].max().to_frame('max_order_size_for_user')
max_os = max_os.reset_index()
max_os.head()


# In[128]:


user = user.merge(max_os, on='user_id', how='left')
user.head()


# In[129]:


del [order_size, avg_os, avg_size, max_os]


# In[130]:


adspo = op.groupby('user_id')['days_since_prior_order'].mean().to_frame('average_days_since_prior_order')
adspo = adspo.reset_index()
adspo.head()


# In[131]:


user = user.merge(adspo, on='user_id', how='left')
user.head()


# In[132]:


del [adspo]


# In[133]:


upu = op.groupby('user_id')['product_id'].apply(f1).to_frame('unique_products_for_user')
upu = upu.reset_index()
upu.head()


# In[134]:


user = user.merge(upu, on='user_id', how='left')
user.head()


# In[136]:


lor = pd.merge(op, user, on='user_id',how='left')
lor = lor[lor.order_number==lor.u_total_orders]
lor.head()


# In[137]:


lorr = lor.groupby('user_id')['reordered'].mean().to_frame('last_order_reordered')
lorr = lorr.reset_index()
lorr.head()


# In[138]:


urr = op.groupby('user_id')['reordered'].mean().to_frame('user_reordered_ratio') #
urr = urr.reset_index()
urr.head()


# In[139]:


user = user.merge(urr, on='user_id', how='left')

del urr
gc.collect()

user.head() 


# In[140]:


ocu = op.groupby(['user_id'])['order_id'].count().to_frame('order_count_for_user')
ocu = ocu.reset_index()
ocu.head()


# In[141]:


user = user.merge(ocu, on='user_id', how='left')
user.head()


# In[142]:


user = user.merge(lorr, on='user_id', how='left')
user.head()


# In[143]:


user_avg_time = op.groupby('user_id')['order_hour_of_day'].mean().to_frame('user_average_time')
user_avg_time = user_avg_time.reset_index()
user_avg_time.head()


# In[144]:


user = user.merge(user_avg_time, on='user_id', how='left')
user.head()


# In[145]:


user_avg_atco = op.groupby('user_id')['add_to_cart_order'].mean().to_frame('user_average_add_to_cart_order')
user_avg_atco = user_avg_atco.reset_index()
user_avg_atco.head()


# In[146]:


user = user.merge(user_avg_atco, on='user_id', how='left')
user.head()


# In[147]:


user_avg_dow = op.groupby('user_id')['order_dow'].mean().to_frame('user_average_day_of_week')
user_avg_dow = user_avg_dow.reset_index()
user_avg_dow.head()


# In[148]:


user = user.merge(user_avg_dow, on='user_id', how='left')
user.head()


# In[149]:


u_atco_median = op.groupby('user_id')['add_to_cart_order'].median().to_frame('user_add_to_cart_order_median') #
u_atco_median = u_atco_median.reset_index()
u_atco_median.head()


# In[150]:


user = user.merge(u_atco_median, on='user_id', how='left')
user.head()


# In[151]:


u_time_median = op.groupby('user_id')['order_hour_of_day'].median().to_frame('user_time_median') #
u_time_median = u_time_median.reset_index()
u_time_median.head()


# In[152]:


user = user.merge(u_time_median, on='user_id', how='left')
user.head()


# In[153]:


u_dspo_median = op.groupby('user_id')['days_since_prior_order'].median().to_frame('user_days_since_prior_median')
u_dspo_median = u_dspo_median.reset_index()
u_dspo_median.head()


# In[154]:


user = user.merge(u_dspo_median, on='user_id', how='left')
user.head()


# In[155]:


del [user_avg_time, user_avg_atco, lorr, ocu]


# In[156]:


uxp = uxp.merge(last_five, on=['user_id', 'product_id'], how='left')

del [last_five]
uxp.head()


# In[157]:


uxp = uxp.fillna(0)
uxp.head()


# In[158]:


data = uxp.merge(user, on='user_id', how='left')
data.head()


# In[159]:


data = data.merge(prr, on='product_id', how='left')
data.head()


# In[160]:


del op, op5, user, prr, uxp, ad
gc.collect()


# In[176]:


## First approach:
# In two steps keep only the future orders from all customers: train & test 
orders_future = orders[((orders.eval_set=='train') | (orders.eval_set=='test'))]
orders_future = orders_future[ ['user_id', 'eval_set', 'order_id'] ]
orders_future.head(10)


# In[177]:


data = data.merge(orders_future, on='user_id', how='left')
data.head(10)


# In[178]:


#Keep only the customers who we know what they bought in their future order
data_train = data[data.eval_set=='train'] #
data_train.head()


# In[179]:


#Get from order_products_train all the products that the train users bought bought in their future order
data_train = data_train.merge(order_products_train[['product_id','order_id', 'reordered']], on=['product_id','order_id'], how='left' )
data_train.head(15)


# In[180]:


#Where the previous merge, left a NaN value on reordered column means that the customers they haven't bought the product. We change the value on them to 0.
data_train['reordered'] = data_train['reordered'].fillna(0)
data_train.head(15)


# In[181]:


#We set user_id and product_id as the index of the DF
data_train = data_train.set_index(['user_id', 'product_id'])
data_train.head(15)


# In[182]:


#We remove all non-predictor variables
data_train = data_train.drop(['eval_set', 'order_id', 'aisle_id', 'department_id'], axis=1)
data_train.head(15)


# In[183]:


data_test = data[data.eval_set=='test'] #
data_test.head()


# In[184]:


#We set user_id and product_id as the index of the DF
data_test = data_test.set_index(['user_id', 'product_id']) #
data_test.head()


# In[185]:


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


paramGrid = {'max_depth':[4,5],
             'learning_rate':[0.07,0.08]
             }

##############
## INSTANTIATE XGBClassifier()####
########################################
xgbc = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', gpu_id=0, tree_method= 'gpu_hist', min_child_weight=1, colsample_bytree=0.9, subsample=0.8, n_estimators=1000)

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

