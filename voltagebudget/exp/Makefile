SHELL=/bin/bash -O expand_aliases
# DATADIR=/home/ejp/src/voltagebudget/voltagebudget/data
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


# ---------------------------------------------------------------------------
# Testing of shadow mode....
# Which should be the focus of the first paper draft? 
# 
# Leave handling spiking hard resets to another day? This comes with a limit of
# the approach no longer bieng compatible with network simulations
# and limits it's immediate applicibility to emperical testing, without
# a VG Na block that is.
#
# Still, it makes it easier to study coordination tradeoffs over whatever T
# and is a clear theoretical extension of prior work in the shadows.
# ---------------------------------------------------------------------------
 
# --
# Stimuli...
# high rate, short duration
# TODO same as sfn
stim1:
	python budget_experiments.py create_stim $(DATADIR)/stim1 0.4 \
	--stim_number=40 --stim_onset=0.2 --stim_offset=0.250 --stim_rate=8 \
	--seed_stim=7525 --time_step=1e-5 --verbose
            

# low rate, continous
# TODO - 1-2 hz..., no specific stim onset.
stim2:
	python budget_experiments.py create_stim $(DATADIR)/stim2 0.4 \
	--stim_number=40 --stim_onset=0.0 --stim_offset=0.4 --stim_rate=2 \
	--seed_stim=754 --time_step=1e-5 --verbose


# -
# Budgets locked to stim1 (stim onset design), and first population spike
# regular firing mode
exp1:
	parallel -j 10 -v \
		--joblog '$(DATADIR)/exp1.log' \
		--nice 19 --delay 2 \
		'python budget_experiments.py forward_shadow $(DATADIR)/exp1_d{1}_f{2} $(DATADIR)/stim1.csv 0.2086 --mode=regular --N=25 --verbose --d={1} --f0={2}' \
		::: 2e-3 4e-3 -6e-3 -8e-3 -10e-3 ::: 4 10 20 40 60 80
 

 # -
# Budgets locked to stim2 (ongoing sparse firing design), arb. middle of the sequence
# regular firing mode
exp2:
	parallel -j 10 -v \
		--joblog '$(DATADIR)/exp2.log' \
		--nice 19 --delay 2 \
		'python budget_experiments.py forward_shadow $(DATADIR)/exp2_d{1}_f{2} $(DATADIR)/stim2.csv 0.1999 --mode=regular --N=25 --verbose --d={1} --f0={2}' ::: \
		2e-3 4e-3 -6e-3 -8e-3 -10e-3 ::: 4 10 20 40 60 80

