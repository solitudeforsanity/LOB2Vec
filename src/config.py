# Parameters - Modifiable for Various Configurations
embedding_size = 120
batch_size = 80
no_epochs = 10
num_frames = 50
Y_len = 8
nb_st_classes = 21
nb_mt_classes = 41
start_side = 0
end_side = 3
start_action = 3
end_action = 10
start_price_level = 10
end_price_level = 27
start_liquidity = 27
end_liquidity = 41
h = 10
w = 2
d = 1
stock_list = ['CCL_NASDAQ.npy', 'EBAY_NASDAQ.npy', 'GIS_NASDAQ.npy', 'SJM_NASDAQ.npy', 'USM_NASDAQ.npy']
#features_classification = ['side', 'action', 'price_level', 'liquidity']
#features_regression = ['mid', 'price', 'spread']

features_classification = ['price_level']
features_regression = ['mid', 'price']
