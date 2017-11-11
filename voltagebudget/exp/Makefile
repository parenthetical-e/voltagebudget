SHELL=/bin/bash -O expand_aliases
# DATADIR=/Users/erikpeterson/Code/fluxen/data
# DATADIR=/Users/type/Code/fluxen/data
# DATADIR=/home/ubuntu/src/fluxen/data
DATADIR=/Users/type/Code/voltagebudget/voltagebudget/data

# 8 Hz
sfn1:
	python budget_experiments.py forward $(DATADIR)/sfn1 --f=8 --save_only --verbose --M=50 --combine_budgets --fix_w

# 4 Hz
sfn2:
	python budget_experiments.py forward $(DATADIR)/sfn2 --f=4 --save_only --verbose --M=50 --combine_budgets --fix_w

# --------------------------
# Modes

# 8 Hz, mode=adaption
sfn3:
	python budget_experiments.py forward $(DATADIR)/sfn3 --f=8 --save_only --verbose --M=50 --combine_budgets --mode=adaption --fix_w


# 8 Hz, mode=adaption
sfn4:
	python budget_experiments.py forward $(DATADIR)/sfn4 --f=8 --save_only --verbose --M=50 --combine_budgets --mode=initial_burst --fix_w 


# 8 Hz, mode=adaption
sfn5:
	python budget_experiments.py forward $(DATADIR)/sfn5 --f=8 --save_only --verbose --M=50 --combine_budgets --mode=regular_burst --fix_w


# 8 Hz, mode=adaption
sfn6:
	python budget_experiments.py forward $(DATADIR)/sfn6 --f=8 --save_only --verbose --M=50 --combine_budgets --mode=delayed_accelerating --fix_w

# --------------------------
# Delays, with regular firing, 8 Hz
sfn7:
	python budget_experiments.py forward $(DATADIR)/sfn7 --f=8 --save_only --verbose --M=50 --combine_budgets --mode=regular --fix_w --budget_delay=-8e-3

sfn8:
	python budget_experiments.py forward $(DATADIR)/sfn8 --f=8 --save_only --verbose --M=50 --combine_budgets --mode=regular --fix_w --budget_delay=-6e-3

sfn9:
	python budget_experiments.py forward $(DATADIR)/sfn9 --f=8 --save_only --verbose --M=50 --combine_budgets --mode=regular --fix_w --budget_delay=-4e-3

sfn10:
	python budget_experiments.py forward $(DATADIR)/sfn10 --f=8 --save_only --verbose --M=50 --combine_budgets --mode=regular --fix_w --budget_delay=-2e-3