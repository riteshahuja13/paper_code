import time
import json
import sys
import pickle
import os
import datetime
import re
import math
import itertools
from functools import partial
from typing import Generator, Mapping, Tuple, NamedTuple, Sequence

import numpy as np
import pandas as pd

import haiku as hk
import jax
from jax import value_and_grad, jit, random
from jax.experimental import optimizers
import jax.numpy as jnp
import optax

# Models
from jax_model import Phi
from jax_model import ResidualStack, ConvEncoder, ConvDecoder, VQVAEModel
from jax_model import D_Encoder, D_Decoder, D_VariationalAutoEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Grid and query processors
from utils import Log, MAE, SAE, mse_weighted_loss, calc_metrics, dist_in_meters, convert_db_to_ijk, gatherresults
from utils import get_train_test_GRID, debiasresultsbreakdown , get_all_cells, _smoothing_constants
from utils import get_queries_for_forecasting, evaluate_forecasting_query, get_hotspot_queries, fast_answer_hostspot_queries
from utils import aggregate_grid, recover_original_grid

model = None

with open('conf.json') as f:
    config = json.load(f) 
    max_val = config['MAX_VAL']
    dim = config['in_dim']
    test_size = config['test_size']
    eps = config['eps']
    rho = config['rho']
    max_data_size = config['n']
    test_query_loc = config['test_query_loc']
    q_w_loc = config['q_w_loc']
    data_loc = config['data_loc']
    int_query_size = np.array(config['int_query_size'])
    lr = config['lr']
    epochs = config['EPOCHS']
    random_seed = config['random_seed']
    exp_comment = config['NAME']
    temporal_stacking = config['temporal_stacking']
    enable_ulevel = config['enable_ulevel']
    base_ulvl = config['base_ulvl']
    base_ulvl_size = config['base_ulvl_size']
    vae_hidden_dim = config['vae_hidden_dim']
    num_snaps = config['num_snaps']
    enable_hostspot_query = config['enable_hostspot_query']
    enable_forecasting_query = config['enable_forecasting_query']
    debiasing_c_ = config['debiasing_c_']
    vq_batchsize = config['vq_batchsize']
    base_granularity = config['base_granularity']
    squeeze = config['squeeze']
    qs_aug = config['qs_aug']

rho = max_val/float(base_granularity) # cell width
augmented_query_size = int_query_size*rho # query size in continuous normalized plane

vae_type = 'none'
if temporal_stacking == 'TS0':
    raise NotImplementedError()
elif temporal_stacking == 'TS_VQCONV':
    vae_type = 'VQconv'

if np.any(int_query_size != 1): # disable auxiliary metrics when query size is larger
    print('###########################Disabling all auxiliary metrics since query is larger than a cell.################################')
    enable_hostspot_query = 0
    enable_forecasting_query = 0

# read data from file and convert it into -5 to 5 catesian range projection.
db = np.load(data_loc)
_D = db.shape[1]
in_dim = _D

db = db[np.isfinite(db)] # are there any nans in data ?
db = db.reshape(-1, _D)

db = db[:max_data_size]
# print('db', db[:5])
print('len(db)', len(db))
print(db[:5])
# for cabs each slice is one day.
min_vals = np.min(db, axis=0)
max_vals = np.max(db, axis=0)
max_vals[2]=max_vals[2]+ (squeeze-1)*(max_vals[2]-min_vals[2])
db = ((db-min_vals)/(max_vals-min_vals)-0.5)*max_val
n = db.shape[0]

if squeeze != 1:
    num_snaps = np.rint(576/squeeze).astype(int)
print('======================== NUM SNAPS ', num_snaps, 'according to squeeze', squeeze, '========================')


print("Data dimensions", db.shape, "min_vals ", min_vals , 'max_vals ', max_vals)
if 1:
    max_temp = (max_vals[2] - min_vals[2] )/ (60*60)
    max_lat_dist = dist_in_meters(max_vals[0], max_vals[1], max_vals[0], min_vals[1])
    max_lon_dist = dist_in_meters(min_vals[0], max_vals[1], max_vals[0], max_vals[1])
    print(f'Data size: {max_temp} hours in the temp. {max_lat_dist} meters in lon. {max_lon_dist} meters in lat.')
    print(f'Approx query size: {(max_temp*augmented_query_size[2]/10)} hours in temp. {max_lat_dist*augmented_query_size[0]/10} meters in lon. {max_lon_dist*augmented_query_size[1]/10} meters in lat.')

test_loc, test_res, identity_test_ress, H, grid_val_noisy, test_loc_ijk, test_loc_ijk_ru, test_loc_ijk_unfiltered = get_train_test_GRID(data_loc,
                                                                                                                                          rho, 
                                                                                                                                          max_val, 
                                                                                                                                          augmented_query_size, 
                                                                                                                                          test_size, 
                                                                                                                                          dim, 
                                                                                                                                          eps, 
                                                                                                                                          test_query_loc, 
                                                                                                                                          db, 
                                                                                                                                          min_vals, 
                                                                                                                                          max_vals, 
                                                                                                                                          random_seed,
                                                                                                                                          q_w_loc, 
                                                                                                                                          enable_ulevel, 
                                                                                                                                          base_ulvl, 
                                                                                                                                          base_ulvl_size, 
                                                                                                                                          num_snaps)


# Cross Check that data filled slices are correct
hunh = True
data_filled_slices = 0
for i in range(0, H.shape[2]):
    # print('(',i, np.sum(H[...,i]), ')', end=' ')
    if hunh and np.sum(H[...,i]) == 0:
        hunh = False
        data_filled_slices = i-1

print("There are ", data_filled_slices, 'data filled slices.')

# Compute Identity noise
_mae = np.average(np.abs(identity_test_ress - test_res), axis=0)[0]
_re = []
for _sm in _smoothing_constants:
    _re.append(np.average(np.abs(identity_test_ress - test_res)/(np.maximum(test_res, _sm)), axis=0)[0])
metrics = [MAE(), SAE(_smoothing_constants[0], "rel. error")]


if enable_forecasting_query:
    forecasting_horizon = 6
    fcast_qs, H_mae, H_mape = get_queries_for_forecasting(data_loc, max_val, min_vals, max_vals, rho,test_size, H, data_filled_slices, forecasting_horizon)
    res_str_H = 'forecasting_H,'+ "{:.4f}".format(H_mae)+ ','+ "{:.4f}".format(H_mape)+ '\n'
    print(res_str_H, end='')
    H_mae = []
    H_mape = []
    _G = grid_val_noisy.copy()
    _G[_G<0] = 0
    for _c in fcast_qs:
        ts = _G[ _c[0], _c[1]].flatten() #grab the tube
        ts_true = H[ _c[0], _c[1]].flatten() #grab the tube
        mae, mape = evaluate_forecasting_query(ts, ts_true, _c[2], forecasting_horizon, forecasting_horizon)
        H_mae.append(mae);H_mape.append(mape)
        # print('len(ts)', len(ts), _c, np.sum(ts), mae, mape)
    H_mae = np.mean(H_mae);H_mape = np.mean(H_mape)
    res_str_G = 'forecasting_G,'+ "{:.4f}".format(H_mae)+ ','+ "{:.4f}".format(H_mape)+ '\n'
    print(res_str_G, end='')

    logfile = open("forecasting_results.log", "a")  # append mode 
    logfile.write(res_str_H)
    logfile.write(res_str_G)
    logfile.close()

Hc_slow_payload = {}
Hress_slow_payload = {}
H_slow_qs = {}
hotspot_query = [10, 20, 40]
hotspot_results = []

if enable_hostspot_query:

    start =time.time()
    for _hot in hotspot_query:
        hotspot_queries, H_c, H_ress = get_hotspot_queries(_hot, H, test_loc_ijk_unfiltered)
        Hc_slow_payload[_hot] = np.array(H_c)
        Hress_slow_payload[_hot] = np.array(H_ress).reshape(-1,1)
        H_slow_qs[_hot] = np.array(hotspot_queries)
        print('Found ' , len(hotspot_queries), 'for', _hot, 'hotness', hotspot_queries[:5])
        _c_er = 0
        ID_c = []
        _hotspot_queries = H_slow_qs[_hot]
        for _loc in _hotspot_queries:
            lb = _loc-50; ru = _loc+50+1;
            lb = np.minimum(np.maximum(0,lb), grid_val_noisy.shape); ru = np.minimum(np.maximum(0,ru), grid_val_noisy.shape);
            _ans_c = fast_answer_hostspot_queries(_hot, lb, ru, grid_val_noisy, _loc)
            ID_c.append(_ans_c)
            if H[tuple(_ans_c)] < _hot:
                _c_er += _hot - H[tuple(_ans_c)]
        ID_c = np.array(ID_c)
        IDQ_res = np.linalg.norm(ID_c - _hotspot_queries, axis=-1).reshape(-1,1)
        HQ_res = Hress_slow_payload[_hot]
        # print('IDQ_res', IDQ_res[:3], 'HQ_res', HQ_res[:3], 'IDQ_res', IDQ_res.shape, 'HQ_res', HQ_res.shape)
        hot_mae = np.average(np.abs(IDQ_res - HQ_res), axis=0)[0]
        hot_re = np.average(np.abs(IDQ_res - HQ_res)/(np.maximum(HQ_res, 1)), axis=0)[0]
        hot_reg = _c_er/max(len(ID_c),1)
        res_str = 'hot3d_ID,'+str(_hot)+ ','+ "{:.4f}".format(hot_mae)+ ','+ "{:.4f}".format(hot_re)+ ','+ "{:.4f}".format(hot_reg)+'\n'
        print(res_str, end='')
        logfile = open("hotspot_results.log", "a")  # append mode 
        logfile.write(res_str)
        logfile.close()

    print("Hotspot slow queries processed in ", time.time()-start, 'seconds')


cum_duration = 0
logs = Log()
hotspot_logs = Log()
fcast_logs = Log()

if enable_ulevel:# get queries and answers from file
    # amp = base_ulvl_size/n
    n_ = n
    N_ = base_ulvl_size
    m_ = np.prod(grid_val_noisy.shape)
    sigma_sq_ = 2/(eps**2)
    c_ = debiasing_c_
    amp = n_*N_*c_ / ( (m_*sigma_sq_) + (1-c_)*n_ + (c_*n_*n_) )
    print('amp_factor_adjusted', amp)
    print('amp_factor_basic', base_ulvl_size/n)
    queries_sfile = data_loc.split('/')
    data_source = queries_sfile[-1][:-4] + '_'
    data_source = data_source+str('_'.join(str(e) for e in int_query_size))
    whatu = re.search('random(.+?)u', data_source).group(1)
    data_source = data_source.replace('random'+str(whatu), 'random' +str(base_ulvl))
    queries_sfile = '/tank/users/ritesh/GHDNH/qfiles/' + data_source +'_'+str(squeeze)+ '_qfile.npy'
    print("Base data source: ", queries_sfile)
    qs_from_file = np.load(queries_sfile)
    splts =  np.hsplit(qs_from_file, [3])
    _locs = splts[0]
    _ans = splts[1]

    # remove queries that are outside of num_snaps range
    _snap_idx = _locs[:, 2] < (num_snaps*rho-5-augmented_query_size[2])
    _locs = _locs[_snap_idx]
    _ans = _ans[_snap_idx]

    print('Number of queries inside snap ', np.sum(_snap_idx))
    print('_locs[:5]', _locs[:5], '_ans[:5]', _ans[:5])

    # generate query bounds
    answer_len = np.rint(augmented_query_size/rho).astype(int)
    _locs_ijk = convert_db_to_ijk(_locs, rho, max_val)  # convert to cell
    _locs_ijk_ru = tuple(np.array(list(zip(*_locs_ijk))) + answer_len)

    # answer queries and compute error
    ulevel_bg_testress = []
    id_ulevel_bg_testress = []
    for x,y in zip(zip(*_locs_ijk), zip(*zip(*_locs_ijk_ru))):
        cell_list = get_all_cells(x,y)
        _val = 0
        _val_noisy = 0
        for c in cell_list:
            _val += H[c]
            _val_noisy += grid_val_noisy[c]
        ulevel_bg_testress.append(_val)
        id_ulevel_bg_testress.append(_val_noisy)
    ulevel_bg_testress = np.array(ulevel_bg_testress).reshape(-1, 1)
    id_ulevel_bg_testress = np.array(id_ulevel_bg_testress).reshape(-1, 1)
    id_ulevel_bg_testress[id_ulevel_bg_testress<0] = 0

    mae, base_mae_debias, re_arr, re_debiased_arr = gatherresults(ulevel_bg_testress, _ans, amp, n)   
    id_mae, id_base_mae_debias, id_re_arr, id_re_debiased_arr = gatherresults(id_ulevel_bg_testress, _ans, amp, n)

    # # log each result in format {alg, dataset, dataset_size, q_range, dist_thres, eps}
    run_result_no_noise = "base_err_no_noise,"+"{:.4f}".format(mae) +","+ str(','.join("{:.4f}".format(e) for e in re_arr)) +","+ "{:.4f}".format(base_mae_debias)+","+ str(','.join("{:.4f}".format(e) for e in re_debiased_arr)) +"\n"
    run_result_identity = "base_err_identity,"+"{:.4f}".format(id_mae) +","+ str(','.join("{:.4f}".format(e) for e in id_re_arr)) +","+ "{:.4f}".format(id_base_mae_debias)+","+ str(','.join("{:.4f}".format(e) for e in id_re_debiased_arr)) +"\n"
    print(run_result_no_noise, run_result_identity)

    result_base_breakdown = 'debias_breakdown'+ ',' + str(','.join("{:.4f}".format(e) for e in mae_breakdown)) + ','+ str(','.join("{:.4f}".format(e) for e in debiased_mae_breakdown)) +"\n"
    result_id_breakdown = 'id_debias_breakdown' + ',' + str(','.join("{:.4f}".format(e) for e in id_mae_breakdown)) + ',' + str(','.join("{:.4f}".format(e) for e in id_debiased_mae_breakdown)) +"\n"
    print(result_base_breakdown, result_id_breakdown)

    if os.path.exists("other_results.log"):
        os.remove("other_results.log")
    logfile = open("other_results.log", "a")  # append mode 
    logfile.write(run_result_no_noise)
    logfile.write(run_result_identity)
    logfile.write(result_base_breakdown)
    logfile.write(result_id_breakdown)
    logfile.close()

def answer_queries_from_H(_H_pred, __test_loc_ijk, __test_loc_ijk_ru):
    _vae_res = []
    for x,y in zip(__test_loc_ijk, __test_loc_ijk_ru):
        cell_list = get_all_cells(x,y)
        _val = 0
        for c in cell_list:
            _val += _H_pred[c]
        _vae_res.append(_val)
    _vae_res = np.array(_vae_res).reshape(-1, 1)
    return _vae_res

_min_base_mae = 1000
def gather_hotspot_results(hotspot_logs, reco_grid, out_str):
    ret_arr = []

    if enable_hostspot_query:
        for _hot in hotspot_query:
            _c_er = 0
            ID_c = []
            _hotspot_queries = H_slow_qs[_hot]
            for _loc in _hotspot_queries:
                lb = _loc-50; ru = _loc+50+1;
                lb = np.minimum(np.maximum(0,lb), reco_grid.shape); ru = np.minimum(np.maximum(0,ru), reco_grid.shape);
                _ans_c = fast_answer_hostspot_queries(_hot, lb, ru, reco_grid, _loc)
                ID_c.append(_ans_c)
                if H[tuple(_ans_c)] < _hot:
                    _c_er += _hot - H[tuple(_ans_c)]
            ID_c = np.array(ID_c)
            IDQ_res = np.linalg.norm(ID_c - _hotspot_queries, axis=-1).reshape(-1,1)
            HQ_res = Hress_slow_payload[_hot]
            # print('IDQ_res', IDQ_res[:3], 'HQ_res', HQ_res[:3], 'IDQ_res', IDQ_res.shape, 'HQ_res', HQ_res.shape)
            hot_mae = np.average(np.abs(IDQ_res - HQ_res), axis=0)[0]
            hot_reg = _c_er/max(len(ID_c),1)
            out_str += '\nhot3d_mae'+str(_hot) +": " +"{:.4f}".format(hot_mae) +" hot3d_reg"+str(_hot)+" : " +"{:.4f}".format(hot_reg) +" "
            # print(out_str)
            hotspot_logs.add('hot3d_mae'+str(_hot), hot_mae)
            hotspot_logs.add('hot3d_reg'+str(_hot), hot_reg)
            ret_arr.append(hot_mae)
    return ret_arr, out_str

def gather_forecasting_results(fcast_logs, reco_grid, out_str):
    H_mae = []
    H_mape = []
    for _c in fcast_qs:
        ts = reco_grid[ _c[0], _c[1]].flatten() #grab the tube
        ts_true = H[ _c[0], _c[1]].flatten() #grab the tube
        mae, mape = evaluate_forecasting_query(ts, ts_true, _c[2], forecasting_horizon, forecasting_horizon)
        H_mae.append(mae);H_mape.append(mape)
        # print('len(ts)', len(ts), _c, np.sum(ts), mae, mape)
    H_mae = np.mean(H_mae);H_mape = np.mean(H_mape)
    out_str += ' FMAE: ' +"{:.4f}".format(H_mae) +" FMAPE: " +"{:.4f}".format(H_mape) +" "
    fcast_logs.add('FMAPE', H_mape)
    return H_mae, H_mape, out_str

_min_mae = 100000
_min_re = [np.inf]*len(_smoothing_constants)

if temporal_stacking == 'TS_VQCONV':
    def gather_standard_results(logs, train_reconstructions, loss_value, out_str):
        vae_ress = answer_queries_from_H(train_reconstructions, test_loc_ijk, test_loc_ijk_ru)
        vae_mae = np.mean(np.abs(vae_ress - test_res), axis=0)[0]
        out_str += " Loss: " + "{:.8f}".format(loss_value)+" "
        out_str += 'vae_mae : ' +"{:.4f}".format(vae_mae) + " id_mae : " +"{:.4f}".format(_mae)
        logs.add("loss", loss_value)
        logs.add("vae_mae", vae_mae)
        logs.add("id_mae", _mae)

        vae_re = []
        for i, _sm in enumerate(_smoothing_constants):
            smre = np.mean(np.abs(vae_ress - test_res)/(np.maximum(test_res, _sm)), axis=0)[0]
            vae_re.append(smre)
            logs.add("vae_re"+str(_sm), smre)
            out_str+=' vae_re'+str(_sm) +" : " +"{:.4f}".format(smre) +" "

            logs.add("id_re"+str(_sm), _re[i])
            out_str +=" id_re"+str(_sm)+" : " +"{:.4f}".format(_re[i]) +" "

        return vae_mae, vae_re, out_str

    train_reconstructions = []

    _slices_per_batch = vq_batchsize
    if num_snaps%_slices_per_batch != 0:
        raise RuntimeError("Must divide batch size")
    no_vq_batches = int(num_snaps/_slices_per_batch) # day batches

    if qs_aug == 'none':
        force_base_grid_as_source = 1
        if force_base_grid_as_source or not (np.any(int_query_size != 1)):
            _train_h = grid_val_noisy.copy()
            _train_h = _train_h[:,:,:num_snaps]
            # _train_h[_train_h<0] = 0
            print('Zeroing Identity negatives')
        else:
            # squeeze over query size
            agg_degree = int_query_size[0]
            print('Squeezing base grid to query size at aggreegation level', agg_degree)
            _train_h = []
            for i in np.arange(num_snaps):
                _train_h.append( aggregate_grid( grid_val_noisy[...,i],degree=agg_degree) )
            _train_h = np.stack(_train_h, axis=2)
            # _train_h[_train_h<0] = 0
            _train_h[_train_h>np.max(H)] = np.max(H) 
        print("Learning from compressed shape", _train_h.shape)
    
    elif qs_aug == 'multi':
        _train_h = grid_val_noisy.copy()
        _train_h = _train_h[:,:,:num_snaps]

        agg_degree = 2
        # print('Squeezing base grid to query size at aggreegation level', agg_degree)
        agg_grid = []
        for i in np.arange(num_snaps):
            agg_grid.append( aggregate_grid( grid_val_noisy[...,i],degree=agg_degree) )
        agg_grid = np.stack(agg_grid, axis=2)
        # print('agg_grid.shape', agg_grid.shape)
        _agg_reco = []
        for i in range(0, num_snaps):
            # print(agg_grid[...,i].shape)
            _reco = recover_original_grid(np.expand_dims(agg_grid[...,i], axis=2) , grid_val_noisy.shape, degree=agg_degree)
            # print(_reco.shape)
            _agg_reco.append(_reco)
        _agg_reco = np.concatenate(_agg_reco, axis=-1)
        # print('_agg_reco.shape', _agg_reco.shape)
        _train_h = np.concatenate([_train_h, _agg_reco], axis=2)
        # print('_train_h.shape', _train_h.shape)
        agg_degree = 4
        # print('Squeezing base grid to query size at aggreegation level', agg_degree)
        agg_grid = []
        for i in np.arange(num_snaps):
            agg_grid.append( aggregate_grid( grid_val_noisy[...,i],degree=agg_degree) )
        agg_grid = np.stack(agg_grid, axis=2)
        # print('agg_grid.shape', agg_grid.shape)
        _agg_reco = []
        for i in range(0, num_snaps):
            # print(agg_grid[...,i].shape)
            _reco = recover_original_grid(np.expand_dims(agg_grid[...,i], axis=2) , grid_val_noisy.shape, degree=agg_degree)
            # print(_reco.shape)
            _agg_reco.append(_reco)
        _agg_reco = np.concatenate(_agg_reco, axis=-1)
        _train_h = np.concatenate([_train_h, _agg_reco], axis=2)
        no_vq_batches *= 3
        # print('Zeroing Identity negatives')
    else:
        raise RuntimeError('qs_aug not recognized :', qs_aug)

    print('_train_h Mean', np.mean(_train_h.flatten()), 'max', np.max(_train_h.flatten()),  'min', np.min(_train_h.flatten()), 'Percentiles [10, 25, 50, 75, 95]', np.percentile(_train_h.flatten(), [10, 25, 50, 75, 95]) )
    
    qt = None
    enableMINMAX = 0
    if enableMINMAX:
        saved_shape = _train_h.shape
        qt = MinMaxScaler((np.min(_train_h.flatten()), np.percentile(_train_h.flatten(), 95)))
        _train_h = qt.fit_transform(_train_h.reshape(-1, 1))
        _train_h = _train_h.reshape(saved_shape) 
        print('_train_h after MINMAX Mean', np.mean(_train_h.flatten()), 'max', np.max(_train_h.flatten()),  'min', np.min(_train_h.flatten()), 'Percentiles [10, 25, 50, 75, 90]', np.percentile(_train_h.flatten(), [10, 25, 50, 75, 90]) )

    train_data_variance = np.var(_train_h.flatten())
    print('train_data_variance', train_data_variance)

    _train_h = np.rollaxis(_train_h, 0, _train_h.ndim)
    _train_h = np.rollaxis(_train_h, 0, _train_h.ndim)
    _train_h = np.expand_dims(_train_h, axis=3)
    print('_train_h.shape', _train_h.shape, 'each batch has shape', _train_h[:_slices_per_batch,...].shape)

    # MODEL Parameters
    num_hiddens = 64
    num_residual_hiddens = 32
    num_residual_layers = 1
    embedding_dim = 64 # smaller would be better for downstream task, maybe set to 24 for the hour of the day ??
    num_embeddings = vae_hidden_dim # The higher this value, the higher the capacity in the information bottleneck.
    commitment_cost = 0.25
    vq_use_ema = True
    decay = 0.99

    def forward(data, is_training):
        encoder = ConvEncoder(num_hiddens, num_residual_layers, num_residual_hiddens)
        decoder = ConvDecoder(num_hiddens, num_residual_layers, num_residual_hiddens)
        pre_vq_conv1 = hk.Conv2D(output_channels=embedding_dim, kernel_shape=(1, 1), stride=(1, 1),name="to_vq")
        if vq_use_ema:
            vq_vae = hk.nets.VectorQuantizerEMA(embedding_dim=embedding_dim,num_embeddings=num_embeddings,commitment_cost=commitment_cost, decay=decay)
        else:
            vq_vae = hk.nets.VectorQuantizer(embedding_dim=embedding_dim,num_embeddings=num_embeddings,commitment_cost=commitment_cost)
        model = VQVAEModel(encoder, decoder, vq_vae, pre_vq_conv1, data_variance=train_data_variance)

        return model(data, is_training)

    forward = hk.transform_with_state(forward)
    optimizer = optax.adam(lr)

    @jax.jit
    def train_step(params, state, opt_state, data):
        def adapt_forward(params, state, data):
            # Pack model output and state together.
            model_output, state = forward.apply(params, state, None, data, is_training=True)
            loss = model_output['loss']
            return loss, (model_output, state)

        grads, (model_output, state) = (jax.grad(adapt_forward, has_aux=True)(params, state, data))
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return params, state, opt_state, model_output

    train_losses = []
    train_recon_errors = []
    train_perplexities = []
    train_vqvae_loss = []

    rng = jax.random.PRNGKey(11)
    params, state = forward.init(rng, _train_h[:_slices_per_batch,...] , is_training=True)
    opt_state = optimizer.init(params)

    print("Training VQCONV for epochs ",int(epochs/2), 'batches', no_vq_batches)

    start =time.time()
    for _e in range(0, int(epochs/2)):
        iters =  no_vq_batches
        for i in range(0, no_vq_batches):
            params, state, opt_state, train_results = (train_step(params, state, opt_state, _train_h[i*_slices_per_batch:(i+1)*_slices_per_batch,...]))
            train_results = jax.device_get(train_results)
            train_losses.append(train_results['loss'])
            train_recon_errors.append(train_results['recon_error'])
            train_perplexities.append(train_results['vq_output']['perplexity'])
            train_vqvae_loss.append(train_results['vq_output']['loss'])

        if _e%10==0:

            train_reconstructions = []
            for i in range(0, int(num_snaps/_slices_per_batch)):
                _train_reco = np.array(forward.apply(params, state, rng, _train_h[i*_slices_per_batch:(i+1)*_slices_per_batch,...], is_training=False)[0]['x_recon']).squeeze()
                _train_reco = np.rollaxis(_train_reco, 0, _train_reco.ndim).squeeze()
                if qs_aug=='none' and not force_base_grid_as_source and (np.any(int_query_size != 1)):
                    _train_reco = recover_original_grid(_train_reco, grid_val_noisy.shape, degree=agg_degree)
                train_reconstructions.append(_train_reco)
            train_reconstructions = np.concatenate(train_reconstructions, axis=-1)

            if enableMINMAX:
                saved_shape = train_reconstructions.shape
                train_reconstructions = qt.inverse_transform(train_reconstructions.reshape(-1,1))
                train_reconstructions = np.reshape(train_reconstructions, saved_shape)

            loss_value = np.mean(train_losses[-iters:])
            vae_mae, vae_re, out_str = gather_standard_results(logs, train_reconstructions, loss_value, str(_e))
            out_str += " time : " +str(time.time()-start) +" "
            if enable_ulevel:
                base_vae_mae, base_vae_re, out_str = gather_ulvl_results(logs, train_reconstructions, out_str)
            if enable_hostspot_query and _e > 50:
                ret_arr, out_str = gather_hotspot_results(hotspot_logs, train_reconstructions, out_str)
                hotspot_logs.save(path='hotspot_results.json')
            if enable_forecasting_query: 
                fmae, fmape, out_str = gather_forecasting_results(fcast_logs, train_reconstructions, out_str)
                fcast_logs.save(path='fcast_results.json')

            print(out_str)
            logs.save(path='vae_results.json')

            print(f'[Step {_e}] ' + ('train loss: %f ' % np.mean(train_losses[-iters:])) +('recon_error: %.3f ' % np.mean(train_recon_errors[-iters:])) + ('perplexity: %.3f ' % np.mean(train_perplexities[-iters:])) +('vqvae loss: %.5f' % np.mean(train_vqvae_loss[-iters:])))

    f = plt.figure(figsize=(16,8))
    ax = f.add_subplot(1,2,1)
    ax.plot(train_recon_errors)
    ax.set_yscale('log')
    ax.set_title('NMSE.')

    ax = f.add_subplot(1,2,2)
    ax.plot(train_perplexities)
    ax.set_title('Average codebook usage (perplexity).')
    f.savefig('conv_vae_losses.png')

