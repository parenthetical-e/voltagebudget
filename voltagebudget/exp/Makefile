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
	--seed_stim=7525 --time_step=1e-5 --verbose


# --
# Stim onset with high rate (stim1)
# Delays and bounds
paper1:
	parallel -j 10 -v \
		--joblog '$(DATADIR)/paper1.log' \
		--nice 19 --delay 2 \
		'python budget_experiments.py forward_shadow $(DATADIR)/paper1 $(DATADIR)/stim1 --M=500 --t=0.4 --f=8 --T_onset=0.2 --T_offset=0.25 --d={1} --budget_reduce_fb={2} --save_only --verbose --combine_budgets --fix_w' ::: 0 2e-3 4e-3 -6e-3 -8e-3 -9e-3 -10e-3 ::: min max mean
 