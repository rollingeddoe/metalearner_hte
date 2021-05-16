# metalearner_hte

## Introduction

This is the repository for our project replicating and extending the contents in [Künzel et al.](https://www.pnas.org/content/pnas/early/2019/02/14/1804597116.full.pdf)

In the replication part we built our own meta-learners based on the pseudo-code in the paper and some ideas from the [causalML package by Uber](https://github.com/uber/causalml) and the [causalToolbox package](https://github.com/soerenkuenzel/hte)

In the extension part we experimented with new confident interval methods based on [bca](https://blogs.sas.com/content/iml/2017/07/12/bootstrap-bca-interval.html) and [bootstrap_t](https://mikelove.wordpress.com/2010/02/15/bootstrap-t/)


The direction of the files are as below:
metalearners.py:  The meta-learners class including S, T, X learners
synthetic_data.py: The data generator for simulations with different distributions
simulations.ipynb: Simulations with 6 different distribution, initialization is based on metalearners.py and synthetic_data.py
 
