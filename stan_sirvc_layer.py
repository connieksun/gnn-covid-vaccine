import csv
import os

import numpy as np
import pandas as pd
#from owid_downloader import GenerateTrainingData
#from utils import date_today, gravity_law_commute_dist

os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '8'

import pickle
import matplotlib.pyplot as plt
import dgl
import torch
from torch import nn
import torch.nn.functional as F
from model import STAN

import sklearn
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

def prepare_data(data, sum_I, sum_R, Vt, edges_df, start, loc_list, history_window=14, pred_window=14, slide_step=3):
    # Data shape n_loc, timestep, n_feat
    # Reshape to n_loc, t, history_window*n_feat
    n_loc = data.shape[0]
    timestep = data.shape[1]
    n_feat = data.shape[2]
    
    x = []
    y_I = []
    y_R = []
    last_I = []
    last_R = []
    concat_I = []
    concat_R = []
    concat_Vt = []
    edges = []
    for i in range(0, timestep, slide_step):
        if i+history_window+pred_window-1 >= timestep or i+history_window >= timestep:
            break
        x.append(data[:, i:i+history_window, :].reshape((n_loc, history_window*n_feat)))
        
        concat_I.append(data[:, i+history_window-1, 0])
        concat_R.append(data[:, i+history_window-1, 1])
        last_I.append(sum_I[:, i+history_window-1])
        last_R.append(sum_R[:, i+history_window-1])

        y_I.append(data[:, i+history_window:i+history_window+pred_window, 0])
        y_R.append(data[:, i+history_window:i+history_window+pred_window, 1])

        concat_Vt.append(Vt[:, i+history_window:i+history_window+pred_window])

        if edges_df is not None:
            e_matrix = np.zeros((n_loc, n_loc)) # edge weight matrix for every time period 
            df = edges_df.groupby(["origin_country"])
            for loc in range(n_loc):
                try:
                    src_df = df.get_group(loc_list[loc])
                    src_df = src_df.groupby(["destination_country"])
                    for loc2 in range(n_loc):
                        try:
                            dst_df = src_df.get_group(loc_list[loc2])
                            dst_df = dst_df.loc[(dst_df['day'] >= (pd.to_datetime(start) + pd.DateOffset(days=i)))]
                            dst_df = dst_df.loc[(dst_df['day'] < (pd.to_datetime(start) + pd.DateOffset(days=i+history_window-1)))]
                            e_matrix[loc, loc2] = dst_df['flight_count'].sum()
                        except:
                            continue
                except:
                    continue
            # normalize matrix (doubly stochastic, see https://arxiv.org/pdf/1809.02709.pdf)
            # step 1: row normalize
            norm = np.sum(e_matrix, axis=1, keepdims=True)
            norm[norm==0] = 1e-10
            norm = 1.0 / norm
            P = e_matrix * norm

            # step 2: P @ P^T / column_norm
            norm = np.sum(P, axis=0, keepdims=True)
            norm[norm==0] = 1e-10
            norm = 1.0 / norm

            PT = np.transpose(P, (1, 0))
            P = np.multiply(P, norm)
            T = np.matmul(P, PT)
            edges.append(T) # n_rows = # countries, n_cols = # countries
        
    
    x = np.array(x, dtype=np.float32).transpose((1, 0, 2))
    last_I = np.array(last_I, dtype=np.float32).transpose((1, 0))
    last_R = np.array(last_R, dtype=np.float32).transpose((1, 0))
    concat_I = np.array(concat_I, dtype=np.float32).transpose((1, 0))
    concat_R = np.array(concat_R, dtype=np.float32).transpose((1, 0))
    y_I = np.array(y_I, dtype=np.float32).transpose((1, 0, 2))
    y_R = np.array(y_R, dtype=np.float32).transpose((1, 0, 2))
    concat_Vt = np.array(concat_Vt, dtype=np.float32).transpose((1, 0, 2))
    return x, last_I, last_R, concat_I, concat_R, y_I, y_R, concat_Vt, edges

def squish_edges(E, rows, cols):
    # use src and dst (rows and cols) to make E matrix
    edges = []
    for M in E:
        edges_flat = []
        for i in range(len(rows)):
            edges_flat.append(M[rows[i]][cols[i]])
        edges.append(edges_flat)
    return np.array(edges)

def train_adaptive_loss(pred_I, pred_I_sir, pred_R, pred_R_sir, true_I, true_R, pred_window=14):
    total_loss = 0
    for timestep in range(len(pred_I)):
        for day in range(1, pred_window+1):
            sir_weight = day/(pred_window+1)
            pred_weight = 1-sir_weight
            sir_loss = sir_weight*((pred_I_sir[timestep][day-1] - true_I[timestep][day-1])**2) + sir_weight*((pred_R_sir[timestep][day-1] - true_R[timestep][day-1])**2)
            pred_loss = pred_weight*((pred_I[timestep][day-1] - true_I[timestep][day-1])**2) + pred_weight*((pred_R[timestep][day-1] - true_R[timestep][day-1])**2)
            total_loss += sir_loss + pred_loss
    return total_loss / pred_window

def val_adaptive_loss(pred_I, pred_I_sir, true_I, pred_window=14):
    total_loss = 0
    for day in range(1, pred_window+1):
        sir_weight = day/(pred_window+1)
        pred_weight = 1-sir_weight
        sir_loss = sir_weight*((pred_I_sir[day-1] - true_I[day-1])**2)
        pred_loss = pred_weight*((pred_I[day-1] - true_I[day-1])**2)
        total_loss += sir_loss + pred_loss
    return total_loss / pred_window


def get_features(raw_data, start_date, end_date, loc_list, edges=True, sum=False):
    # Generate Graph
    # add flight neighbors
    # for now, add a connection if there is any flight between the two countries between start and end date
    loc_list = list(raw_data['location'].unique())
    flight_counts = pd.read_csv('processed_flights/flight_counts_2021_all_to_05.csv')
    adj_map = {}
    for each_loc in loc_list:
        df = flight_counts.loc[flight_counts["origin_country"] == each_loc]
        adj_map[each_loc] = set(df["destination_country"].unique())
    flight_counts['day'] = pd.to_datetime(flight_counts['day'])
    # add land neighbors
    neighbor_reader = csv.reader(open('neighbors.csv', 'r'))
    neighbors = {}
    for row in neighbor_reader:
        neighbors[row[0]] = row[1].split(',')
    for each_loc,connected in adj_map.items():
        for neighbor in neighbors[each_loc]:
            if neighbor in loc_list:
                connected.add(neighbor)
    # create graph
    rows = []
    cols = []
    for each_loc in adj_map:
        for each_loc2 in adj_map[each_loc]:
            if each_loc in loc_list and each_loc2 in loc_list:
                rows.append(loc_list.index(each_loc))
                cols.append(loc_list.index(each_loc2))
    #print(rows)
    #print(cols)
    g = dgl.graph((rows, cols))
    print(g.number_of_nodes)
    #Preprocess features

    #active_cases = []
    confirmed_cases = []
    new_cases = []
    new_vaccinations = []
    fully_vaccinated = []
    death_cases = []
    static_feat = []

    for i, each_loc in enumerate(loc_list):
        confirmed_cases.append(raw_data[raw_data['location'] == each_loc]['total_cases'])
        new_cases.append(raw_data[raw_data['location'] == each_loc]['new_cases_smoothed'])
        new_vaccinations.append(raw_data[raw_data['location'] == each_loc]['new_vaccinations'])
        fully_vaccinated.append(raw_data[raw_data['location'] == each_loc]['people_fully_vaccinated'])
        death_cases.append(raw_data[raw_data['location'] == each_loc]['total_deaths'])
        static_feat.append(np.array(raw_data[raw_data['location'] == each_loc][['population']]))
    confirmed_cases = np.nan_to_num(np.array(confirmed_cases))
    death_cases = np.nan_to_num(np.array(death_cases)[:, 14:])
    #new_cases = np.nan_to_num(np.array(new_cases)[:, 14:])
    new_cases = np.nan_to_num(np.array(new_cases))
    new_vaccinations = np.nan_to_num(np.array(new_vaccinations)[:, 14:])
    fully_vaccinated = np.nan_to_num(np.array(fully_vaccinated))
    static_feat = np.nan_to_num(np.array(static_feat)[:, 0, :])

    # total_cases = []
    # for i,loc in enumerate(confirmed_cases):
    #     total_loc = [loc[0]]
    #     for j in range(1, len(loc)):
    #         total_loc.append(total_loc[-1] + new_cases[i][j])
    #     total_cases.append(total_loc)
    # total_cases = np.array(total_cases)

    # confirmed_cases = total_cases

    import copy
    # active = confirmed(today) - confirmed(14 days ago)
    cases_copy = copy.deepcopy(confirmed_cases)
    active = []
    for loc in confirmed_cases:
        active_loc = []
        for i in range(14, len(loc)):
            active_loc.append(loc[i] - loc[i-14])
        active.append(active_loc)
    active_cases = np.array(active)

    confirmed_cases = confirmed_cases[:, 14:]
    fully_vaccinated = fully_vaccinated[:, 14:]

    recovered_cases = confirmed_cases - active_cases - death_cases + 0.94*fully_vaccinated
    susceptible_cases = np.expand_dims(static_feat[:, 0], -1) - active_cases - recovered_cases

    # Batch_feat: new_cases(dI), dR, dS
    #dI = np.array(new_cases)
    dI = np.concatenate((np.zeros((active_cases.shape[0],1), dtype=np.float32), np.diff(active_cases)), axis=-1)
    dR = np.concatenate((np.zeros((recovered_cases.shape[0],1), dtype=np.float32), np.diff(recovered_cases)), axis=-1)
    dS = np.concatenate((np.zeros((susceptible_cases.shape[0],1), dtype=np.float32), np.diff(susceptible_cases)), axis=-1)
    # number of new fully vaccinated each day
    Vt = np.concatenate((np.zeros((fully_vaccinated.shape[0],1), dtype=np.float32), np.diff(fully_vaccinated)), axis=-1)
    print("done")

    #Build normalizer
    normalizer = {'S':{}, 'I':{}, 'R':{}, 'dS':{}, 'dI':{}, 'dR':{}, 'Vt':{}}

    for i, each_loc in enumerate(loc_list):
        normalizer['S'][each_loc] = (np.mean(susceptible_cases[i]), np.std(susceptible_cases[i]))
        normalizer['I'][each_loc] = (np.mean(active_cases[i]), np.std(active_cases[i]))
        normalizer['R'][each_loc] = (np.mean(recovered_cases[i]), np.std(recovered_cases[i]))
        normalizer['dI'][each_loc] = (np.mean(dI[i]), np.std(dI[i]))
        normalizer['dR'][each_loc] = (np.mean(dR[i]), np.std(dR[i]))
        normalizer['dS'][each_loc] = (np.mean(dS[i]), np.std(dS[i]))
        normalizer['Vt'][each_loc] = (np.mean(Vt[i]), np.std(Vt[i]))

    valid_window = 29
    test_window = 29

    history_window=14 # days of information
    pred_window=14 # predicts future # of days
    slide_step=3 # increment

    dynamic_feat = np.concatenate((np.expand_dims(dI, axis=-1), np.expand_dims(dR, axis=-1), np.expand_dims(dS, axis=-1)), axis=-1)
        
    #Normalize
    for i, each_loc in enumerate(loc_list):
        dynamic_feat[i, :, 0] = (dynamic_feat[i, :, 0] - normalizer['dI'][each_loc][0]) / normalizer['dI'][each_loc][1]
        dynamic_feat[i, :, 1] = (dynamic_feat[i, :, 1] - normalizer['dR'][each_loc][0]) / normalizer['dR'][each_loc][1]
        dynamic_feat[i, :, 2] = (dynamic_feat[i, :, 2] - normalizer['dS'][each_loc][0]) / normalizer['dS'][each_loc][1]
        # vaccinations don't need to be normalized
        #mean_vax = normalizer['Vt'][each_loc][0]
        #if mean_vax != 0:
        #   Vt[i] = (Vt[i] - mean_vax) / normalizer['Vt'][each_loc][1]
    dI_mean = []
    dI_std = []
    dR_mean = []
    dR_std = []

    for i, each_loc in enumerate(loc_list):
        dI_mean.append(normalizer['dI'][each_loc][0])
        dR_mean.append(normalizer['dR'][each_loc][0])
        dI_std.append(normalizer['dI'][each_loc][1])
        dR_std.append(normalizer['dR'][each_loc][1])

    dI_mean = np.array(dI_mean)
    dI_std = np.array(dI_std)
    dR_mean = np.array(dR_mean)
    dR_std = np.array(dR_std)

    #Split train-test
    train_feat = dynamic_feat[:, :-valid_window-test_window, :]
    val_feat = dynamic_feat[:, -valid_window-test_window:-test_window, :]
    test_feat = dynamic_feat[:, -test_window:, :]

    valid_start_date = pd.to_datetime(end_date) + pd.DateOffset(days=-valid_window) + pd.DateOffset(days=-test_window)
    test_start_date = pd.to_datetime(end_date) + pd.DateOffset(days=-test_window)

    if edges:
        train_edges = flight_counts[(flight_counts["day"] >= start_date) & (flight_counts["day"] < valid_start_date)]
        val_edges = flight_counts[(flight_counts["day"] >= valid_start_date) & (flight_counts["day"] < test_start_date)]
        test_edges = flight_counts[(flight_counts["day"] >= test_start_date) & (flight_counts["day"] < end_date)]
    else:
        train_edges, val_edges, test_edges = None, None, None

    if sum:
        active_cases = np.expand_dims(active_cases.sum(axis=0), axis=0)
        recovered_cases = np.expand_dims(recovered_cases.sum(axis=0), axis=0)
        susceptible_cases = np.expand_dims(susceptible_cases.sum(axis=0), axis=0)
        fully_vaccinated = np.expand_dims(fully_vaccinated.sum(axis=0), axis=0)

        Vt = np.concatenate((np.zeros((fully_vaccinated.shape[0],1), dtype=np.float32), np.diff(fully_vaccinated)), axis=-1)
        loc_list = [0]

        dI = np.concatenate((np.zeros((active_cases.shape[0],1), dtype=np.float32), np.diff(active_cases)), axis=-1)
        dR = np.concatenate((np.zeros((recovered_cases.shape[0],1), dtype=np.float32), np.diff(recovered_cases)), axis=-1)
        dS = np.concatenate((np.zeros((susceptible_cases.shape[0],1), dtype=np.float32), np.diff(susceptible_cases)), axis=-1)
        # number of new fully vaccinated each day
        Vt = np.concatenate((np.zeros((fully_vaccinated.shape[0],1), dtype=np.float32), np.diff(fully_vaccinated)), axis=-1)
        print("done")

        #Build normalizer
        normalizer = {'S':{}, 'I':{}, 'R':{}, 'dS':{}, 'dI':{}, 'dR':{}, 'Vt':{}}

        for i, each_loc in enumerate(loc_list):
            normalizer['S'][each_loc] = (np.mean(susceptible_cases[i]), np.std(susceptible_cases[i]))
            normalizer['I'][each_loc] = (np.mean(active_cases[i]), np.std(active_cases[i]))
            normalizer['R'][each_loc] = (np.mean(recovered_cases[i]), np.std(recovered_cases[i]))
            normalizer['dI'][each_loc] = (np.mean(dI[i]), np.std(dI[i]))
            normalizer['dR'][each_loc] = (np.mean(dR[i]), np.std(dR[i]))
            normalizer['dS'][each_loc] = (np.mean(dS[i]), np.std(dS[i]))
            normalizer['Vt'][each_loc] = (np.mean(Vt[i]), np.std(Vt[i]))

        dynamic_feat = np.concatenate((np.expand_dims(dI, axis=-1), np.expand_dims(dR, axis=-1), np.expand_dims(dS, axis=-1)), axis=-1)
            
        #Normalize
        for i, each_loc in enumerate(loc_list):
            dynamic_feat[i, :, 0] = (dynamic_feat[i, :, 0] - normalizer['dI'][each_loc][0]) / normalizer['dI'][each_loc][1]
            dynamic_feat[i, :, 1] = (dynamic_feat[i, :, 1] - normalizer['dR'][each_loc][0]) / normalizer['dR'][each_loc][1]
            dynamic_feat[i, :, 2] = (dynamic_feat[i, :, 2] - normalizer['dS'][each_loc][0]) / normalizer['dS'][each_loc][1]

        dI_mean = []
        dI_std = []
        dR_mean = []
        dR_std = []

        for i, each_loc in enumerate(loc_list):
            dI_mean.append(normalizer['dI'][each_loc][0])
            dR_mean.append(normalizer['dR'][each_loc][0])
            dI_std.append(normalizer['dI'][each_loc][1])
            dR_std.append(normalizer['dR'][each_loc][1])

        dI_mean = np.array(dI_mean)
        dI_std = np.array(dI_std)
        dR_mean = np.array(dR_mean)
        dR_std = np.array(dR_std)

    train_x, train_I, train_R, train_cI, train_cR, train_yI, train_yR, train_Vt, train_edges = prepare_data(train_feat, active_cases[:, :-valid_window-test_window], recovered_cases[:, :-valid_window-test_window], Vt[:, :-valid_window-test_window], train_edges, start_date, loc_list, history_window, pred_window, slide_step)

    val_x, val_I, val_R, val_cI, val_cR, val_yI, val_yR, val_Vt, val_edges = prepare_data(val_feat, active_cases[:, -valid_window-test_window:-test_window], recovered_cases[:, -valid_window-test_window:-test_window], Vt[:, -valid_window-test_window:-test_window], val_edges, valid_start_date, loc_list, history_window, pred_window, slide_step)

    test_x, test_I, test_R, test_cI, test_cR, test_yI, test_yR, test_Vt, test_edges = prepare_data(test_feat, active_cases[:, -test_window:], recovered_cases[:, -test_window:], Vt[:, -test_window:], test_edges, test_start_date, loc_list, history_window, pred_window, slide_step)

    if edges:
        train_edges = squish_edges(train_edges, rows, cols)
        val_edges = squish_edges(val_edges, rows, cols)
        test_edges = squish_edges(test_edges, rows, cols)

        print(train_edges.shape) # one edge array (len = # edges) for each timestep
        print(train_x.shape) # one array of features for each timestep for each location
        print(val_edges.shape)
        print(val_x.shape)
        print(test_edges.shape)
        print(test_x.shape)

    all_features = []
    all_features.append([train_x, train_I, train_R, train_cI, train_cR, train_yI, train_yR, train_Vt, train_edges])
    all_features.append([val_x, val_I, val_R, val_cI, val_cR, val_yI, val_yR, val_Vt, val_edges])
    all_features.append([test_x, test_I, test_R, test_cI, test_cR, test_yI, test_yR, test_Vt, test_edges])

    norms = [dI_mean, dI_std, dR_mean, dR_std]

    return g, all_features, active_cases, static_feat, norms


def run_layer(g, features, active_cases, static_feat, norms, start_date, end_date, loc_list, loc_name, trained=False):

    valid_window = 29
    test_window = 29

    history_window=14 # days of information
    pred_window=14 # predicts future # of days
    slide_step=3 # increment

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train = features[0]
    train_x = torch.tensor(train[0]).to(device)
    train_I = torch.tensor(train[1]).to(device)
    train_R = torch.tensor(train[2]).to(device)
    train_cI = torch.tensor(train[3]).to(device)
    train_cR = torch.tensor(train[4]).to(device)
    train_yI = torch.tensor(train[5]).to(device)
    train_yR = torch.tensor(train[6]).to(device)
    train_Vt = torch.tensor(train[7]).to(device)
    train_edges = torch.tensor(train[8]).to(device)

    val = features[1]
    val_x = torch.tensor(val[0]).to(device)
    val_I = torch.tensor(val[1]).to(device)
    val_R = torch.tensor(val[2]).to(device)
    val_cI = torch.tensor(val[3]).to(device)
    val_cR = torch.tensor(val[4]).to(device)
    val_yI = torch.tensor(val[5]).to(device)
    val_yR = torch.tensor(val[6]).to(device)
    val_Vt = torch.tensor(val[7]).to(device)
    val_edges = torch.tensor(val[8]).to(device)

    test = features[2]
    test_x = torch.tensor(test[0]).to(device)
    test_I = torch.tensor(test[1]).to(device)
    test_R = torch.tensor(test[2]).to(device)
    test_cI = torch.tensor(test[3]).to(device)
    test_cR = torch.tensor(test[4]).to(device)
    test_yI = torch.tensor(test[5]).to(device)
    test_yR = torch.tensor(test[6]).to(device)
    test_Vt = torch.tensor(test[7]).to(device)
    test_edges = torch.tensor(test[8]).to(device)

    dI_mean = torch.tensor(norms[0], dtype=torch.float32).to(device).reshape((norms[0].shape[0], 1, 1))
    dI_std = torch.tensor(norms[1], dtype=torch.float32).to(device).reshape((norms[0].shape[0], 1, 1))
    dR_mean = torch.tensor(norms[2], dtype=torch.float32).to(device).reshape((norms[0].shape[0], 1, 1))
    dR_std = torch.tensor(norms[3], dtype=torch.float32).to(device).reshape((norms[0].shape[0], 1, 1))

    N = torch.tensor(static_feat[:, 0], dtype=torch.float32).to(device).unsqueeze(-1)
    in_dim = 3*history_window
    hidden_dim1 = 32
    hidden_dim2 = 32
    gru_dim = 32
    num_heads = 1

    g = g.to(device)
    model = STAN(g, in_dim, hidden_dim1, hidden_dim2, gru_dim, num_heads, pred_window, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    cur_loc = loc_list.index(loc_name)  
    
    # build STAN model 
    if not trained:

        #Train STAN

        all_loss = []
        file_name = './all_models/stan_' + loc_name
        min_loss = 1e20

        for epoch in range(60):
            model.train()
            optimizer.zero_grad()
            
            active_pred, recovered_pred, phy_active, phy_recover, _ = model(train_x, train_cI[cur_loc], train_cR[cur_loc], N[cur_loc], train_I[cur_loc], train_R[cur_loc], V=train_Vt[cur_loc], e_weights=train_edges)
            phy_active = (phy_active - dI_mean[cur_loc]) / dI_std[cur_loc]
            phy_recover = (phy_recover - dR_mean[cur_loc]) / dR_std[cur_loc]
            # SIR loss = (day) / (pred_window + 1); dynamics loss = 1 - SIR loss 
            #loss = criterion(active_pred.squeeze(), train_yI[cur_loc].squeeze())+criterion(recovered_pred.squeeze(), train_yR[cur_loc].squeeze()) \
            #    + 0.1*criterion(phy_active.squeeze(), train_yI[cur_loc].squeeze())+0.1*criterion(phy_recover.squeeze(), train_yR[cur_loc].squeeze())
            loss = train_adaptive_loss(active_pred.squeeze(), phy_active.squeeze(), recovered_pred.squeeze(), phy_recover.squeeze(), train_yI[cur_loc].squeeze(), train_yR[cur_loc].squeeze(), pred_window)

            loss.backward()
            optimizer.step()
            all_loss.append(loss.item())
            
            model.eval()
            _, _, _, _, prev_h = model(train_x, train_cI[cur_loc], train_cR[cur_loc], N[cur_loc], train_I[cur_loc], train_R[cur_loc], V=train_Vt[cur_loc], e_weights=train_edges)
            val_active_pred, val_recovered_pred, val_phy_active, val_phy_recover, _ = model(val_x, val_cI[cur_loc], val_cR[cur_loc], N[cur_loc], val_I[cur_loc], val_R[cur_loc], prev_h, V=val_Vt[cur_loc], e_weights=val_edges)
            
            val_phy_active = (val_phy_active - dI_mean[cur_loc]) / dI_std[cur_loc]
            # SIR loss = (day) / (pred_window + 1); dynamics loss = 1 - SIR loss 
            # change loss here 
            #val_loss = criterion(val_active_pred.squeeze(), val_yI[cur_loc].squeeze()) + 0.1*criterion(val_phy_active.squeeze(), val_yI[cur_loc].squeeze())
            val_loss = val_adaptive_loss(val_active_pred.squeeze(), val_phy_active.squeeze(), val_yI[cur_loc].squeeze(), pred_window)
            #if val_loss < min_loss: 
            if (val_loss + loss) / 2 < min_loss:   
                state = {
                    'state': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, file_name)
                #min_loss = val_loss
                min_loss = (val_loss + loss) / 2
                print('-----Save best model-----')
            
            print('Epoch %d, Loss %.2f, Val loss %.2f'%(epoch, all_loss[-1], val_loss.item()))

    #Pred with STAN
    file_name = './all_models/stan_' + loc_name
    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint['state'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()

    prev_x = torch.cat((train_x, val_x), dim=1)
    prev_I = torch.cat((train_I, val_I), dim=1)
    prev_R = torch.cat((train_R, val_R), dim=1)
    prev_cI = torch.cat((train_cI, val_cI), dim=1)
    prev_cR = torch.cat((train_cR, val_cR), dim=1)
    prev_Vt = torch.cat((train_Vt, val_Vt), dim=1)
    prev_edges = torch.cat((train_edges, val_edges), dim=0)
    prev_active_pred, _, prev_phy_active_pred, _, h = model(prev_x, prev_cI[cur_loc], prev_cR[cur_loc], N[cur_loc], prev_I[cur_loc], prev_R[cur_loc], V=prev_Vt[cur_loc], e_weights=prev_edges)

    test_pred_active, test_pred_recovered, test_pred_phy_active, test_pred_phy_recover, _ = model(test_x, test_cI[cur_loc], test_cR[cur_loc], N[cur_loc], test_I[cur_loc], test_R[cur_loc], h, V=test_Vt[cur_loc], e_weights=test_edges)

    #_, _, _, _, h = model(train_x, train_cI[cur_loc], train_cR[cur_loc], N[cur_loc], train_I[cur_loc], train_R[cur_loc], V=train_Vt[cur_loc], e_weights=train_edges)

    #test_pred_active, test_pred_recovered, test_pred_phy_active, test_pred_phy_recover, _ = model(val_x, val_cI[cur_loc], val_cR[cur_loc], N[cur_loc], val_I[cur_loc], val_R[cur_loc], h, V=val_Vt[cur_loc], e_weights=val_edges)

    # print('Estimated beta in SIR model is %.2f'%model.alpha_scaled)
    # print('Estimated gamma in SIR model is %.2f'%model.beta_scaled)
    # print('Estimated theta in SIR model is %.2f'%model.theta_scaled)

    pred_I = []

    for i in range(test_pred_active.size(1)):
        # below is regular prediction
        cur_pred = (test_pred_active[0, i, :].detach().cpu().numpy() * dI_std[cur_loc].reshape(1, 1).detach().cpu().numpy()) + dI_mean[cur_loc].reshape(1, 1).detach().cpu().numpy()
        pred_I.append(cur_pred)

    pred_I = np.array(pred_I)

    pred_I_prev = []
    for i in range(prev_active_pred.size(1)):
        # below is regular prediction
        cur_pred = (prev_active_pred[0, i, :].detach().cpu().numpy() * dI_std[cur_loc].reshape(1, 1).detach().cpu().numpy()) + dI_mean[cur_loc].reshape(1, 1).detach().cpu().numpy()
        pred_I_prev.append(cur_pred)

    pred_I_prev = np.array(pred_I_prev)

    return pred_I, pred_I_prev