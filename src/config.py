import os

# Base paths relative to the src directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "intern_data_db")
WAREHOUSE_DIR = os.path.join(BASE_DIR, "warehouse")

# Data folders
SPOT_DIR = os.path.join(DATA_DIR, "nifty_spot")
FUTURES_DIR = os.path.join(DATA_DIR, "nifty_futures")
OPTIONS_DIR = os.path.join(DATA_DIR, "options_chain")
AUX_DIR = os.path.join(DATA_DIR, "aux_data")
if not os.path.exists(AUX_DIR):
    AUX_DIR = os.path.join(DATA_DIR, "aux")
