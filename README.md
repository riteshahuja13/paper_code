# VDR
This repository contains the implementation of VAE Data Release (VDR) to answer range count queries on a spatio-temporal datasets while preserving differential privacy. VQVAE is trained using JAX and used to answer RCQs, HotSpot, and Forecasting queries.

## Instalation and requirements

#### Install conda environment and python targets:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh

#### create a python target and source into python enivorment
conda create -n [name_of_enviornment] python=3.9

conda activate [name_of_enviornment]

#### install jax lib 0.1.71 with CUDA and cuDNN support  (without GPU is also okay, ~20x performance penalty with 18 core CPU) 
wget https://storage.googleapis.com/jax-releases/cuda111/jaxlib-0.1.71+cuda111-cp38-none-manylinux2010_x86_64.whl

pip install jaxlib-0.1.71+cuda111-cp38-none-manylinux2010_x86_64.whl

#### install other libraries
pip install jax==0.2.12 numpy pandas dm-haiku sklearn rtree optax

## Running VDR
Running VDR can be done by calling python run.py. VDR configureations are set in run.py, through the python dictionary config. Specifically, the dictionary contains a key 'NAME'. When calling python run.py, the code creates the folder tests/config['NAME'], where the result of the experiment is written. Explanation of each of the configurations is available in file run.py.

## VDR output
VDR model's training and testing statistic is written in the file tests/config['NAME']/i/out.txt

## Example
The folder data contains two datasets: CABS_SF_SE.npy (from [1]). We consider releasing CABS_SF.npy with differential privacy. Calling python run.py performs training and testing with this setting. For example, the result of the zero-th trained model will be written in tests/test_sf_cabs/0/out.txt. A sample output for that file is 

>Creating model 

cabs 846654 len(settings) 1
{'eps': 0.2, 'int_query_size': [1, 1, 1]}
1
len(db) 846654
======================== NUM SNAPS  24 according to squeeze 24 ========================

Data dimensions (846654, 3) min_vals  [ 3.7607930e+01 -1.2245129e+02  1.2110184e+09] max_vals  [ 3.78079200e+01 -1.22211340e+02  1.26072471e+09]

Data size: 13807.306666666667 hours in the temp. 21103.56261858769 meters in lon. 22262.686809999963 meters in lat.

Approx query size: 23.97101851851852 hours in temp. 36.638129546159185 meters in lon. 38.65049793402771 meters in lat.

[0.01736111 0.01736111 0.01736111] 0.017361111111111112
[1 1 1] [0. 0. 0.]

Histogram min 0  max 61  mean 0.004422746404535323 median 0.0 sum 845200  Percentiles [25,50,75,95,99] [0. 0. 0. 0. 0.]
grid_val_noisy mem size 1458.0

Noisy Histogram min -88.14647912196452  max 92.67523937383527  mean 0.005002342404628596 median 0.002736341412788847

Generated test_loc results 2000 at 12.933814287185669 seconds  min 0  max 54  mean 7.438 median 25tile 75tile:  5.0 2.0 10.0

There are  24 data filled slices.

_train_h Mean 0.1068140995737103 max 77.87652824615176 min -87.69362722079335 Percentiles [10, 25, 50, 75, 95] [-4.47663357 -1.87198272  0.07025465  2.04426261  7.28331811]

train_data_variance 22.462744002789577
_train_h.shape (72, 576, 576, 1) each batch has shape (4, 576, 576, 1)

Training VQCONV for epochs  750 batches 18

0 Loss: 1.07852447 vae_mae : 7.1133 id_mae : 5.0003 vae_re20 : 0.3189  time : 6.809430837631226

10 Loss: 0.89561599 vae_mae : 4.5922 id_mae : 5.0003 vae_re20 : 0.2018  time : 44.396788597106934

20 Loss: 0.82908553 vae_mae : 4.4749 id_mae : 5.0003 vae_re20 : 0.1984  time : 81.97453165054321

30 Loss: 0.77791733 vae_mae : 4.4070 id_mae : 5.0003 vae_re20 : 0.1968  time : 119.54707360267639

40 Loss: 0.75882417 vae_mae : 4.3414 id_mae : 5.0003 vae_re20 : 0.1952  time : 157.09501242637634

50 Loss: 0.74739593 vae_mae : 4.2074 id_mae : 5.0003 vae_re20 : 0.1895  time : 194.62223172187805

60 Loss: 0.73977876 vae_mae : 4.1342 id_mae : 5.0003 vae_re20 : 0.1870  time : 232.1827037334442

70 Loss: 0.73555321 vae_mae : 4.2247 id_mae : 5.0003 vae_re20 : 0.1918  time : 269.7674515247345

80 Loss: 0.71234769 vae_mae : 4.3001 id_mae : 5.0003 vae_re20 : 0.1956  time : 307.305330991745

90 Loss: 0.69417262 vae_mae : 4.2565 id_mae : 5.0003 vae_re20 : 0.1940  time : 344.8118648529053

100 Loss: 0.68460965 vae_mae : 4.1163 id_mae : 5.0003 vae_re20 : 0.1880  time : 382.3187086582184

110 Loss: 0.68375659 vae_mae : 4.0784 id_mae : 5.0003 vae_re20 : 0.1865  time : 419.8280494213104

120 Loss: 0.67941034 vae_mae : 4.0346 id_mae : 5.0003 vae_re20 : 0.1849  time : 457.3497130870819

130 Loss: 0.67818046 vae_mae : 4.2066 id_mae : 5.0003 vae_re20 : 0.1923  time : 494.8889949321747

140 Loss: 0.67680240 vae_mae : 4.0577 id_mae : 5.0003 vae_re20 : 0.1860  time : 532.4169631004333

150 Loss: 0.68126178 vae_mae : 3.9777 id_mae : 5.0003 vae_re20 : 0.1824  time : 569.922712802887

160 Loss: 0.67663527 vae_mae : 4.0975 id_mae : 5.0003 vae_re20 : 0.1883  time : 607.4535751342773

170 Loss: 0.67851412 vae_mae : 3.9950 id_mae : 5.0003 vae_re20 : 0.1837  time : 644.9919319152832

180 Loss: 0.67981434 vae_mae : 4.3227 id_mae : 5.0003 vae_re20 : 0.1984  time : 682.5634868144989

190 Loss: 0.67378974 vae_mae : 4.0437 id_mae : 5.0003 vae_re20 : 0.1860  time : 720.1283006668091

200 Loss: 0.67333502 vae_mae : 3.9890 id_mae : 5.0003 vae_re20 : 0.1837  time : 757.6599857807159

210 Loss: 0.67543018 vae_mae : 3.9525 id_mae : 5.0003 vae_re20 : 0.1822  time : 795.2138011455536

280 Loss: 0.67564249 vae_mae : 3.8169 id_mae : 5.0003 vae_re20 : 0.1795  time : 1058.0666770935059

## References
[1] Michal Piorkowski, Natasa Sarafijanovic-Djukic, and Matthias Grossglauser. 2009.CRAWDAD data set epfl/mobility (v. 2009-02-24)
