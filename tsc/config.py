"""Default settings for traffic-signal-control experiments.

The MPLight values are copied from the official MPLight repository
(github.com/Chacha-Chen/MPLight): ``config.py`` -> DIC_TRANSFERDQNPRESSONE_AGENT_CONF,
``config.py`` -> dic_traffic_env_conf, and the overrides in ``runexp.py``.
Key names are kept identical so every value can be traced back to the source.
"""

from __future__ import annotations

# --- Agent (official: DIC_TRANSFERDQNPRESSONE_AGENT_CONF + runexp.py overrides) ---
MPLIGHT_AGENT_CONF = {
    "LEARNING_RATE": 0.001,     # Adam, epsilon=1e-08
    "SAMPLE_SIZE": 1000,        # samples drawn from memory per round
    "BATCH_SIZE": 20,
    "EPOCHS": 100,              # with early stopping on val_loss
    "PATIENCE": 10,
    "VALIDATION_SPLIT": 0.3,    # keras fit(validation_split=0.3, shuffle=False)
    "UPDATE_Q_BAR_FREQ": 5,     # target net = model saved UPDATE_Q_BAR_FREQ rounds ago
    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 10000,
    "D_DENSE": 20,
    "EPSILON": 0.8,             # epsilon_round = max(EPSILON * EPSILON_DECAY**round, MIN_EPSILON)
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "NORMAL_FACTOR": 20,        # target = reward / NORMAL_FACTOR + GAMMA * max Q_bar(s')
}

# --- Environment (official: dic_traffic_env_conf + runexp.py) ---
ENV_CONF = {
    "MIN_ACTION_TIME": 10,      # seconds per decision (includes yellow)
    "YELLOW_TIME": 5,
    "ALL_RED_TIME": 0,
    "MEASURE_TIME": 10,         # reward averaged over this window
    "RUN_COUNTS": 3600,         # simulated seconds per episode
    "REWARD_PRESSURE_WEIGHT": -0.25,   # DIC_REWARD_INFO["pressure"]
}

# --- Experiment (official: DIC_EXP_CONF + runexp.py) ---
EXP_CONF = {
    "NUM_ROUNDS": 100,          # runexp.py uses 100-700 depending on the dataset
    "NUM_GENERATORS": 1,        # runexp.py --gen default
    "TEST_EVERY": 1,            # official pipeline tests after every round
}

# --- SUMO adaptation (not in MPLight: CityFlow grids do not need these) ---
SUMO_CONF = {
    "CLUSTER_DIST": 80.0,       # m: traffic lights closer than this form one junction/agent
    "ZONE_LENGTH": 200.0,       # m: approach/exit counting zone (CityFlow lanes are 300 m)
    "GLUE_LENGTH": 30.0,        # m: unsignalised node between two parts of one junction
    "PHASE_MODE": "plan",       # "plan": stages of the original signal plans
                                # "conflict": maximal sets of compatible movements
    "MIN_PHASE_RUN": 5,         # s: "plan" mode ignores stages shorter than this
    "SIM_SEED": 0,
}
