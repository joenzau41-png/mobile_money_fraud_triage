# Data

This project uses the **PaySim** synthetic mobile-money dataset (Kaggle:
`ealaxi/paysim1`, ~6.3M rows). **The data is synthetic**, generated to mimic
mobile-money transaction logs. It is not real M-Pesa data.

Check the Kaggle page for the current licence before reuse.

## Get the data

Option A, manual: download the zip from Kaggle, extract the CSV to
`data/raw/PS_20174392719_1491204439457_log.csv`.

Option B, Kaggle API: place `kaggle.json` in `~/.kaggle/`, then run:

    python -m src.data download

## Convert to Parquet (full file + 10% dev sample)

    python -m src.data convert

Raw and processed data are git-ignored and never committed.