#!/bin/bash
# Run eval_subgraphx.py using the dedicated subgraphx_env (Python 3.12 + dgl).
# See requirements_subgraphx.txt for the pinned dependency set.

./subgraphx_env/bin/python3 eval_subgraphx.py \
    --begin 0 \
    --end 10 \
    --sim_index vgae \
    --dataset mutag \
    --kaggle 0 \
    --interp_index learned \
    --max_nodes 10 \
    --eval_stab 0
