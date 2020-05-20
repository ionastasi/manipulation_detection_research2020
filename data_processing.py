import numpy as np
import pandas as pd


def read_messages(filename):
	'''
	simply reads Message Flow from the provided file and returns the df
	'''
    mesg = pd.read_csv(filename, header=None, names=['Time', 
                                                     'Type', 
                                                     'Order_id', 
                                                     'Size', 
                                                     'Price', 
                                                     'Direction'])
    return mesg


def _get_ohlcv(group):
	'''
	helper function. calculates ohlcv statistics over given df
	'''
    return {'Time': group['Time'].iloc[0],
                    'Open': group['Price'].iloc[0], 
                    'High': np.max(group['Price']), 
                    'Low': np.min(group['Price']), 
                    'Close': group['Price'].iloc[-1], 
                    'Volume': np.sum(group['Size'])}


def transform_to_ohlcv(messages):
	'''
	fiven the Message Flow, returns OHLCV aggregated over seconds
	'''
    result = messages.copy()
    result = result[(result['Type'] == 4) | (result['Type'] == 5)]  # filter executed orders
    result = result.groupby(np.floor(result['Time']), as_index=False)
    result = result.apply(_get_ohlcv).apply(pd.Series)

    # add missing seconds
    first_second = int(np.floor(result['Time'].iloc[0]))
    last_second = int(np.ceil(result['Time'].iloc[-1]))
    for idx in range(last_second - first_second):
        cur_sec = int(np.floor(result['Time'].iloc[idx]))
        if cur_sec != first_second + idx: # then it's a missing second
            add_line = result.iloc[[idx]].copy()
            add_line['Time'] = first_second + idx
            result = pd.concat([result.iloc[:idx], add_line, result.iloc[idx:]], ignore_index=True)

    return result



def heuristic(messages, time_start, timewindow):
	'''
	calculates heuristics, returns df witg single pump&dump column
	'''
    group = messages[(messages['Time'] >= time_start) & (messages['Time'] < time_start + timewindow)]

    n_third = group.shape[0] // 3
    part_1 = group.iloc[:n_third, :]
    part_2 = group.iloc[n_third:n_third*2, :]
    part_3 = group.iloc[n_third*2:, :]

    # pumping: increase in price
    threshold_1 = 0.002
    p_matched_min_buy =  part_1[((part_1['Type'] == 4) | (part_1['Type'] == 5)) & (part_1['Direction'] == 1)]['Price'].min()
    p_matched_max_buy =  part_2[((part_2['Type'] == 4) | (part_2['Type'] == 5)) & (part_2['Direction'] == 1)]['Price'].max()
    pump = (p_matched_max_buy - p_matched_min_buy) > (p_matched_max_buy * threshold_1)


    # dumping: decrease in price
    threshold_2 = 0.002
    p_matched_max_sell = part_2[((part_2['Type'] == 4) | (part_2['Type'] == 5)) & (part_2['Direction'] == -1)]['Price'].max()
    p_matched_min_sell = part_3[((part_3['Type'] == 4) | (part_3['Type'] == 5)) & (part_3['Direction'] == -1)]['Price'].min()
    dump_1 = (p_matched_max_sell - p_matched_min_sell) > (p_matched_max_sell * threshold_2)
    
    # dumping: many canceled orders
    threshold_3 = 0.4
    v_cancelled_buy =    part_2[((part_2['Type'] == 2) | (part_2['Type'] == 3)) & (part_2['Direction'] == 1)]['Size'].sum()
    v_matched_buy_mean = part_2[((part_2['Type'] == 4) | (part_2['Type'] == 5)) & (part_2['Direction'] == 1)]['Size'].mean()
    dump_2 = v_cancelled_buy > (v_matched_buy_mean * threshold_3)

    return {'pump&dump': dump_1 & dump_2 & pump}


def label_data(messages, timewindow=60):
    '''
    returns labels
    messages are aggregated by 60 seconds with a sliding step 60/3=20 seconds
    '''
    parts = 3
    shift = timewindow // parts
    shifted_messages = messages.copy()
    shifted_messages['Time'] = np.floor(shifted_messages['Time'] / shift)
    first_second = int(np.floor(shifted_messages['Time'].iloc[0]))
    last_second = int(np.ceil(shifted_messages['Time'].iloc[-1]))
    result = pd.DataFrame({'Time': list(range(first_second, last_second - parts + 1))})
    result['Label'] = result.apply(lambda x: heuristic(shifted_messages, x['Time'], parts), axis=1)
    result = result['Label'].apply(pd.Series)
    result['Time'] = list(range(first_second, last_second - parts + 1))

    return result



def stack_seconds(ohlcv, timeframe=60):
	'''
	stacks 60 consequent OHLCV snapshots to one feature vector
	'''
    eps=1
    ohlcv_cleaned = ohlcv.copy().drop(columns=['Time'])
    ohlcv_cleaned['Log_Volume'] = np.log(ohlcv_cleaned.pop('Volume') + eps)
    ohlcv_cleaned['Log_Open'] = np.log(ohlcv_cleaned.pop('Open') + eps)
    ohlcv_cleaned['Log_High'] = np.log(ohlcv_cleaned.pop('High') + eps)
    ohlcv_cleaned['Log_Low'] = np.log(ohlcv_cleaned.pop('Low') + eps)
    ohlcv_cleaned['Log_Close'] = np.log(ohlcv_cleaned.pop('Close') + eps)
    result = ohlcv_cleaned.copy().add_suffix("_0")
    for sec in range(1, timeframe):
        result = result.join(ohlcv_cleaned.iloc[sec:,].copy().reset_index(drop=True).add_suffix("_{}".format(sec)))
    result = result.drop(result.tail(timeframe-1).index)

    return result


def main(messages_filename='sample_messages.csv', output_filename='aggregated_ohlcv_labeled.csv'):
	messages = read_messages(messages_filename)
	labels = label_data(messages)
	ohlcv = transform_to_ohlcv(messages)
	ohlcv_stacked = stack_seconds(ohlcv)
	ohlcv_stacked['Label'] = labels['pump&dump']
	ohlcv_stacked.to_csv(output_filename)
