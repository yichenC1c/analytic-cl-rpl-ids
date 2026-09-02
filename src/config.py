"""Global configuration.

All values mirror the reference implementation released with the RPL continual-learning
benchmark (Banerjee et al., ICC 2026) so that results are directly comparable:

    utils.py  parse_args()   hyperparameter defaults
    utils.py  load_data()    split and normalisation protocol
    main.py   line 134       LSTMClassifier(fc_hidden_dim=10)
    main.py   lines 149-190  the four domain orderings

The orderings are extracted programmatically into domain_orders.json rather than
transcribed by hand.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "IoT-Attacks-IDS", "src", "attack_data")
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")

# ---------------------------------------------------------------- data protocol
WINDOW_SIZE = 10        # sliding window length used by seq_maker
N_RUNS = 20             # first 20 simulation runs per domain
N_TRAIN_FILES = 16      # 16 train / 4 test, split by run rather than by row
VAL_FILES = 3           # held out from the 16 training runs, hyperparameter search only
SPLIT_SEED = 42         # matches random.seed(42) in the reference loader
N_RAW_FEAT = 14         # 7 RPL attributes x (mean, std)
INPUT_SIZE = N_RAW_FEAT * WINDOW_SIZE   # 140, flattened and fed as a length-1 sequence

# ---------------------------------------------------------------- encoder
HIDDEN_SIZE = 10        # LSTM hidden width
FC_HIDDEN_DIM = 10      # main.py passes 10 explicitly, overriding the default of 64
OUTPUT_SIZE = 2
NUM_LAYERS = 1
HEAD_DROPOUT = 0.3
LEARNING_RATE = 1e-3
BATCH_SIZE = 128        # the reference code uses 128 where its paper states 256
EPOCHS = 100            # upper bound; early stopping usually fires after 25-40
PATIENCE = 2

# ---------------------------------------------------------------- analytic head
# D and RIDGE are chosen by select_hyperparams.py on a validation split of the first
# domain and frozen afterwards. Test data is never loaded during selection.
RFF_D = 64
RFF_GAMMA = None        # None selects the median heuristic on first-domain features
RIDGE = 1e+00
RLS_BLOCK = 16          # Woodbury block size; must not exceed D (see analytic.RLSHead)

SEEDS = [0, 1, 2]

# ---------------------------------------------------------------- domain orderings
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "domain_orders.json")) as f:
    DOMAIN_ORDERS = json.load(f)
SCENARIOS = ["random", "b2w", "w2b", "toggle"]
