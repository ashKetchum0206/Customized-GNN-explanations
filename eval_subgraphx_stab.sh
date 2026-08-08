#!/bin/bash
# Keep the machine from sleeping for the duration of this (potentially
# many-hour) run -- macOS will otherwise suspend every child process the
# moment the system goes to sleep, which silently stalls the whole batch.
# `-i` blocks idle sleep (works on battery too), `-s` blocks system sleep
# (AC power only), `-m` blocks disk sleep. Display sleep is left alone.
if [ -z "$CAFFEINATED" ]; then
    export CAFFEINATED=1
    exec caffeinate -i -s -m "$0" "$@"
fi

# Stability-only SubgraphX evals (--eval_stab 1), kept separate from
# eval_subgraphx.sh (full interp+fidelity sweeps) since stability is ~10x
# more expensive per graph.
#
# mutag (188 graphs) is small enough to run in full. ba2motif, bamultishapes,
# and proteins run on a 200-graph coreset (coreset_<dataset>.npy) selected by
# select_coreset.py: stratified by (predicted class, confidence decile, size
# decile) with a farthest-point-sampling pick within each stratum.

COMMANDS=(
  # --- mutag: full dataset, no coreset ---
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 188 --sim_index gntk --dataset mutag --kaggle 1 --interp_index learned --max_nodes 12 --eval_stab 1"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 188 --sim_index vgae --dataset mutag --kaggle 1 --interp_index learned --max_nodes 12 --eval_stab 1"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 188 --sim_index gntk --dataset mutag --kaggle 1 --interp_index learned --max_nodes 10 --eval_stab 1"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 188 --sim_index vgae --dataset mutag --kaggle 1 --interp_index learned --max_nodes 10 --eval_stab 1"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 188 --sim_index gntk --dataset mutag --kaggle 1 --interp_index learned --max_nodes 8 --eval_stab 1"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 188 --sim_index vgae --dataset mutag --kaggle 1 --interp_index learned --max_nodes 8 --eval_stab 1"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 188 --sim_index gntk --dataset mutag --kaggle 1 --interp_index learned --max_nodes 6 --eval_stab 1"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 188 --sim_index vgae --dataset mutag --kaggle 1 --interp_index learned --max_nodes 6 --eval_stab 1"

  # --- ba2motif: 200-graph coreset ---
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_ba2motif.npy --sim_index gntk --dataset ba2motif --kaggle 1 --interp_index learned --max_nodes 12 --eval_stab 1"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_ba2motif.npy --sim_index vgae --dataset ba2motif --kaggle 1 --interp_index learned --max_nodes 12 --eval_stab 1"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_ba2motif.npy --sim_index gntk --dataset ba2motif --kaggle 1 --interp_index learned --max_nodes 10 --eval_stab 1"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_ba2motif.npy --sim_index vgae --dataset ba2motif --kaggle 1 --interp_index learned --max_nodes 10 --eval_stab 1"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_ba2motif.npy --sim_index gntk --dataset ba2motif --kaggle 1 --interp_index learned --max_nodes 8 --eval_stab 1"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_ba2motif.npy --sim_index vgae --dataset ba2motif --kaggle 1 --interp_index learned --max_nodes 8 --eval_stab 1"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_ba2motif.npy --sim_index gntk --dataset ba2motif --kaggle 1 --interp_index learned --max_nodes 6 --eval_stab 1"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_ba2motif.npy --sim_index vgae --dataset ba2motif --kaggle 1 --interp_index learned --max_nodes 6 --eval_stab 1"

  # --- bamultishapes: 200-graph coreset ---
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_bamultishapes.npy --sim_index gntk --dataset bamultishapes --kaggle 1 --interp_index learned --max_nodes 12 --eval_stab 1"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_bamultishapes.npy --sim_index vgae --dataset bamultishapes --kaggle 1 --interp_index learned --max_nodes 12 --eval_stab 1"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_bamultishapes.npy --sim_index gntk --dataset bamultishapes --kaggle 1 --interp_index learned --max_nodes 10 --eval_stab 1"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_bamultishapes.npy --sim_index vgae --dataset bamultishapes --kaggle 1 --interp_index learned --max_nodes 10 --eval_stab 1"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_bamultishapes.npy --sim_index gntk --dataset bamultishapes --kaggle 1 --interp_index learned --max_nodes 8 --eval_stab 1"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_bamultishapes.npy --sim_index vgae --dataset bamultishapes --kaggle 1 --interp_index learned --max_nodes 8 --eval_stab 1"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_bamultishapes.npy --sim_index gntk --dataset bamultishapes --kaggle 1 --interp_index learned --max_nodes 6 --eval_stab 1"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_bamultishapes.npy --sim_index vgae --dataset bamultishapes --kaggle 1 --interp_index learned --max_nodes 6 --eval_stab 1"

#   # --- proteins: 200-graph coreset ---
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_proteins.npy --sim_index gntk --dataset proteins --kaggle 1 --interp_index learned --max_nodes 12 --eval_stab 1"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_proteins.npy --sim_index vgae --dataset proteins --kaggle 1 --interp_index learned --max_nodes 12 --eval_stab 1"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_proteins.npy --sim_index gntk --dataset proteins --kaggle 1 --interp_index learned --max_nodes 10 --eval_stab 1"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_proteins.npy --sim_index vgae --dataset proteins --kaggle 1 --interp_index learned --max_nodes 10 --eval_stab 1"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_proteins.npy --sim_index gntk --dataset proteins --kaggle 1 --interp_index learned --max_nodes 8 --eval_stab 1"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_proteins.npy --sim_index vgae --dataset proteins --kaggle 1 --interp_index learned --max_nodes 8 --eval_stab 1"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_proteins.npy --sim_index gntk --dataset proteins --kaggle 1 --interp_index learned --max_nodes 6 --eval_stab 1"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_proteins.npy --sim_index vgae --dataset proteins --kaggle 1 --interp_index learned --max_nodes 6 --eval_stab 1"



  # --- mutag: full dataset, no coreset ---
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 188 --sim_index gntk --dataset mutag --kaggle 1 --interp_index learned --max_nodes 12 --eval_stab 0"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 188 --sim_index vgae --dataset mutag --kaggle 1 --interp_index learned --max_nodes 10 --eval_stab 0"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 188 --sim_index gntk --dataset mutag --kaggle 1 --interp_index learned --max_nodes 8 --eval_stab 0"
#   "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 188 --sim_index vgae --dataset mutag --kaggle 1 --interp_index learned --max_nodes 6 --eval_stab 0"

  # --- ba2motif: full dataset ---
  # "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 900 --sim_index gntk --dataset ba2motif --kaggle 1 --interp_index learned --max_nodes 12 --eval_stab 0"
  # "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 900 --sim_index vgae --dataset ba2motif --kaggle 1 --interp_index learned --max_nodes 10 --eval_stab 0"
  # "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 900 --sim_index gntk --dataset ba2motif --kaggle 1 --interp_index learned --max_nodes 8 --eval_stab 0"
  # "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 900 --sim_index vgae --dataset ba2motif --kaggle 1 --interp_index learned --max_nodes 6 --eval_stab 0"

  # --- bamultishapes: full dataset ---
  # "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 900 --sim_index gntk --dataset bamultishapes --kaggle 1 --interp_index learned --max_nodes 12 --eval_stab 0"
  # "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 900 --sim_index vgae --dataset bamultishapes --kaggle 1 --interp_index learned --max_nodes 10 --eval_stab 0"
  # "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 900 --sim_index gntk --dataset bamultishapes --kaggle 1 --interp_index learned --max_nodes 8 --eval_stab 0"
  # "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 900 --sim_index vgae --dataset bamultishapes --kaggle 1 --interp_index learned --max_nodes 6 --eval_stab 0"

#   # --- proteins: full dataset ---
  "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 900 --sim_index gntk --dataset proteins --kaggle 1 --interp_index learned --max_nodes 12 --eval_stab 0"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 900 --sim_index vgae --dataset proteins --kaggle 1 --interp_index learned --max_nodes 10 --eval_stab 0"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 900 --sim_index gntk --dataset proteins --kaggle 1 --interp_index learned --max_nodes 8 --eval_stab 0"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 900 --sim_index vgae --dataset proteins --kaggle 1 --interp_index learned --max_nodes 6 --eval_stab 0"
)

set +e

# --- Configuration ---
# Cap concurrency at the number of PERFORMANCE cores, not total logical
# cores. On Apple Silicon `hw.ncpu` counts P+E cores together (e.g. 4P+4E=8
# on an M2), but E-cores are much slower for this kind of dense numeric
# workload, so oversubscribing onto them just adds contention. There is no
# user-space API to hard-pin a process to a specific physical core on macOS
# (Apple Silicon does not honor the Mach thread-affinity API at all), so the
# only real levers are: (1) how many processes we run at once, and (2) how
# many threads each process is allowed to use internally.
PCORES=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
MAX_PARALLEL=$(( PCORES < 4 ? PCORES : 4 ))

TOTAL_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "$PCORES")
# Each of the MAX_PARALLEL concurrent tasks gets an equal thread budget so
# that torch/numpy/BLAS don't each independently spin up a thread pool sized
# to ALL cores -- eval_subgraphx.py never calls torch.set_num_threads(), so
# without this, 4 concurrent processes x ~8 threads each = 32-way
# oversubscription on an 8-core machine, which is almost certainly why the
# overnight run crawled.
THREADS_PER_TASK=$(( TOTAL_CORES / MAX_PARALLEL ))
[ "$THREADS_PER_TASK" -lt 1 ] && THREADS_PER_TASK=1

echo -e "\n🚀 eval_subgraphx_stab.sh: Starting parallel execution of ${#COMMANDS[@]} tasks (Max concurrent: $MAX_PARALLEL, P-cores detected: $PCORES, threads/task: $THREADS_PER_TASK)\n"

# --- execution loop ---
# NOTE: throttling is done by tracking PIDs and polling `kill -0`, not by
# counting `jobs -r`. On macOS's stock bash (3.2), calling the `jobs`
# builtin from the parent while a backgrounded subshell contains a compound
# command (like the if/else below) forces bash to synchronize on that
# child, which silently serializes the whole loop.
PIDS=()
for i in "${!COMMANDS[@]}"; do
    CMD="${COMMANDS[$i]}"
    JOB_NUM=$((i+1))
    LOG_FILE="task_${JOB_NUM}.log"

    echo "▶️ Launching Task $JOB_NUM/${#COMMANDS[@]} — Logging to $LOG_FILE"

    # Run the command in a background subshell. Thread-limiting env vars are
    # scoped to this subshell only, so they don't leak across concurrent
    # tasks.
    (
        export OMP_NUM_THREADS=$THREADS_PER_TASK
        export MKL_NUM_THREADS=$THREADS_PER_TASK
        export OPENBLAS_NUM_THREADS=$THREADS_PER_TASK
        export VECLIB_MAXIMUM_THREADS=$THREADS_PER_TASK
        export NUMEXPR_NUM_THREADS=$THREADS_PER_TASK

        echo "=== Task Started at $(date) ===" > "$LOG_FILE"
        echo "Command: $CMD" >> "$LOG_FILE"
        echo "Thread budget: $THREADS_PER_TASK" >> "$LOG_FILE"
        echo -e "------   -----------------------------------\n" >> "$LOG_FILE"

        eval "$CMD" >> "$LOG_FILE" 2>&1
        STATUS=$?

        if [ $STATUS -eq 0 ]; then
            echo "✅ Task $JOB_NUM succeeded."
        else
            echo "❌ Task $JOB_NUM failed with exit code $STATUS. See $LOG_FILE"
        fi
    ) &
    PIDS+=($!)
    # Make sure this task isn't left running under Darwin's throttled
    # background QoS tier (nice/priority settings are preserved across the
    # exec() inside `eval`, so this applies to the actual python process too).
    taskpolicy -B -p "$!" >/dev/null 2>&1

    # Throttle control: block until fewer than MAX_PARALLEL tracked PIDs are still alive
    while :; do
        ALIVE=0
        for p in "${PIDS[@]}"; do
            kill -0 "$p" 2>/dev/null && ALIVE=$((ALIVE+1))
        done
        [ "$ALIVE" -lt "$MAX_PARALLEL" ] && break
        sleep 1
    done
done

echo -e "\n⏳ All tasks spawned. Waiting for active background processes to wrap up..."
wait

echo -e "\n🎉 All parallel tasks have completed!"