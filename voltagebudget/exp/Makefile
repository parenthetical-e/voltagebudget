SHELL=/bin/bash -O expand_aliases
DATADIR=/Users/qualia/Code/voltagebudget/data
# DATADIR=/home/ejp/src/voltagebudget/voltagebudget/data
# DATADIR=/Users/type/Code/voltagebudget/voltagebudget/data

# ------------------------------------------------------------------------
# SFN runs are all deprecated. 
# to rerun use 'cb9c43a'
# ------------------------------------------------------------------------

# # 8 Hz
# sfn1:
# 	python budget_experiments.py forward $(DATADIR)/sfn1 --f=8 --save_only --verbose --M=50 --combine_budgets --fix_w

# # 4 Hz
# sfn2:
# 	python budget_experiments.py forward $(DATADIR)/sfn2 --f=4 --save_only --verbose --M=50 --combine_budgets --fix_w

# # --------------------------
# # Modes

# # 8 Hz, mode=adaption
# sfn3:
# 	python budget_experiments.py forward $(DATADIR)/sfn3 --f=8 --save_only --verbose --M=50 --combine_budgets --mode=adaption --fix_w


# # 8 Hz, mode=adaption
# sfn4:
# 	python budget_experiments.py forward $(DATADIR)/sfn4 --f=8 --save_only --verbose --M=50 --combine_budgets --mode=initial_burst --fix_w 


# # 8 Hz, mode=adaption
# sfn5:
# 	python budget_experiments.py forward $(DATADIR)/sfn5 --f=8 --save_only --verbose --M=50 --combine_budgets --mode=regular_burst --fix_w


# # 8 Hz, mode=adaption
# sfn6:
# 	python budget_experiments.py forward $(DATADIR)/sfn6 --f=8 --save_only --verbose --M=50 --combine_budgets --mode=delayed_accelerating --fix_w

# # Delays, with regular firing, 8 Hz
# sfn7:
# 	python budget_experiments.py forward $(DATADIR)/sfn7 --f=8 --save_only --verbose --M=50 --combine_budgets --mode=regular --fix_w --budget_delay=-8e-3

# sfn8:
# 	python budget_experiments.py forward $(DATADIR)/sfn8 --f=8 --save_only --verbose --M=50 --combine_budgets --mode=regular --fix_w --budget_delay=-6e-3

# sfn9:
# 	python budget_experiments.py forward $(DATADIR)/sfn9 --f=8 --save_only --verbose --M=50 --combine_budgets --mode=regular --fix_w --budget_delay=-4e-3

# sfn10:
# 	python budget_experiments.py forward $(DATADIR)/sfn10 --f=8 --save_only --verbose --M=50 --combine_budgets --mode=regular --fix_w --budget_delay=-2e-3


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

# lower rate, many more neurons (default .json requires tweaks for this to run
# with reaslistic firing rates).
stim3:
	python budget_experiments.py create_stim $(DATADIR)/stim3 0.4 \
	--stim_number=1000 --stim_onset=0.0 --stim_offset=0.4 --stim_rate=1.6 \
	--seed_stim=20 --time_step=1e-5 --verbose

# A large N burst. (must reduce w_in -> w_in * 0.2)
# with weak synapses
stim4:
	python budget_experiments.py create_stim $(DATADIR)/stim4 0.4 \
	--stim_number=1000 --stim_onset=0.2 --stim_offset=0.250 --stim_rate=2 \
	--seed_stim=7525 --time_step=1e-5 --verbose

# A small N burst. (must increase min of w_in -> w_in[0] * 2)
stim5:
	python budget_experiments.py create_stim $(DATADIR)/stim5 0.4 \
	--stim_number=10 --stim_onset=0.2 --stim_offset=0.250 --stim_rate=20 \
	--seed_stim=7525 --time_step=1e-5 --verbose

# A long version of stim1
stim6:
	python budget_experiments.py create_stim $(DATADIR)/stim6 7.0 \
	--stim_number=40 --stim_onset=0.2 --stim_offset=6.8 --stim_rate=8 \
	--seed_stim=7525 --time_step=1e-5 --verbose

# ---------------------------------------------------------------------------
# Budgets locked to stim1 (stim onset design), and first population spike

# regular firing mode:
# f=4
exp1:
	parallel -j 4 -v \
		--joblog '$(DATADIR)/exp1.log' \
		--nice 19 --delay 2 \
		'python budget_experiments.py forward $(DATADIR)/exp1_d{1}_f{2} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=50 --d={1} --T=0.25 --f={2} --verbose' \
		::: -2e-3 ::: 4

# f=8
exp2:
	parallel -j 4 -v \
		--joblog '$(DATADIR)/exp2.log' \
		--nice 19 --delay 2 \
		'python budget_experiments.py forward $(DATADIR)/exp2_d{1}_f{2} $(DATADIR)/stim1.csv 0.0 --mode=regular  --A_0=0.05e-9 --A_max=0.2e-9 --N=50 --d={1} --T=0.125 --f={2} --verbose' \
		::: -2e-3 ::: 8 
 
# f=30
exp3:
	parallel -j 4 -v \
		--joblog '$(DATADIR)/exp3.log' \
		--nice 19 --delay 2 \
		'python budget_experiments.py forward $(DATADIR)/exp3_d{1}_f{2} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=50 --d={1} --T=0.033 --f={2} --verbose' \
		::: -2e-3 ::: 30


# Pareto
# TODO set f based on exp1-3
exp4:
	parallel -j 4 -v \
		--joblog '$(DATADIR)/exp4.log' \
		--nice 19 --delay 2 \
		'python budget_experiments.py pareto $(DATADIR)/exp4_d{1}_f{2} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_max=0.05e-9 --N=50 --d={1} --T=0.125 --f={2} --verbose' \
		::: -2e-3 ::: 12 20 30
 
 # Explore single neuron models, by ranks
 # TODO set f based on exp1-3
 exp5:
	parallel -j 4 -v \
		--joblog '$(DATADIR)/exp5.log' \
		--nice 19 --delay 2 \
		'python budget_experiments.py ranked $(DATADIR)/exp5_d{1}_f{2}_r{3} $(DATADIR)/stim1.csv 0.0 --mode=regular  --A_0=0.05e-9 --A_max=0.2e-9 --N=50 --rank={3} --d={1} --T=0.125 --f={2} --verbose' \
		::: -2e-3 ::: 8 ::: 1 2 3 48 49 50

# 1-25-2018
# Run forward on more freqs
# f=12
exp6:
	parallel -j 4 -v \
		--joblog '$(DATADIR)/exp6.log' \
		--nice 19 --delay 2 \
		'python budget_experiments.py forward $(DATADIR)/exp6_d{1}_f{2} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=50 --d={1} --T=0.083 --f={2} --verbose' \
		::: -2e-3 ::: 12

# f=20
exp7:
	parallel -j 4 -v \
		--joblog '$(DATADIR)/exp7.log' \
		--nice 19 --delay 2 \
		'python budget_experiments.py forward $(DATADIR)/exp7_d{1}_f{2} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=50 --d={1} --T=0.05 --f={2} --verbose' \
		::: -2e-3 ::: 20

# f=25
exp8:
	parallel -j 4 -v \
		--joblog '$(DATADIR)/exp8.log' \
		--nice 19 --delay 2 \
		'python budget_experiments.py forward $(DATADIR)/exp8_d{1}_f{2} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=50 --d={1} --T=0.04 --f={2} --verbose' \
		::: -2e-3 ::: 25 

# 1-26-2018
# f=35
exp9:
	parallel -j 4 -v \
		--joblog '$(DATADIR)/exp9.log' \
		--nice 19 --delay 2 \
		'python budget_experiments.py forward $(DATADIR)/exp9_d{1}_f{2} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=50 --d={1} --T=0.029 --f={2} --verbose --save_only' \
		::: -2e-3 ::: 35


# f=40
exp10:
	parallel -j 4 -v \
		--joblog '$(DATADIR)/exp10.log' \
		--nice 19 --delay 2 \
		'python budget_experiments.py forward $(DATADIR)/exp10_d{1}_f{2} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=50 --d={1} --T=0.025 --f={2} --verbose --save_only' \
		::: -2e-3 ::: 40 

# f=50
exp11:
	parallel -j 4 -v \
		--joblog '$(DATADIR)/exp11.log' \
		--nice 19 --delay 2 \
		'python budget_experiments.py forward $(DATADIR)/exp11_d{1}_f{2} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=50 --d={1} --T=0.02 --f={2} --verbose --save_only' \
		::: -2e-3 ::: 50

# f=60
exp12:
	parallel -j 4 -v \
		--joblog '$(DATADIR)/exp12.log' \
		--nice 19 --delay 2 \
		'python budget_experiments.py forward $(DATADIR)/exp12_d{1}_f{2} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=50 --d={1} --T=0.016 --f={2} --verbose --save_only' \
		::: -2e-3 ::: 60

# f=25
exp13:
	parallel -j 4 -v \
		--joblog '$(DATADIR)/exp13.log' \
		--nice 19 --delay 2 \
		'python budget_experiments.py forward $(DATADIR)/exp13_d{1}_f{2} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=50 --d={1} --T=0.04 --f={2} --verbose --save_only' \
		::: -2e-3 ::: 25 

# ------------------------------------------------------------------
# 1-26-2018
# More single unit models
# Explore f space (adj T to match)
 exp14:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp14.log' \
		--nice 19 --delay 2 \
		'python budget_experiments.py ranked $(DATADIR)/exp14_d{1}_f{2}_r{3} $(DATADIR)/stim1.csv 0.0 --mode=regular  --A_0=0.05e-9 --A_max=0.2e-9 --N=50 --rank={3} --d={1} --T=0.25 --f={2} --verbose --save_only' \
		::: -2e-3 ::: 4 ::: 1 2 3 48 49 50

exp15:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp15.log' \
		--nice 19 --delay 2 \
		'python budget_experiments.py ranked $(DATADIR)/exp15_d{1}_f{2}_r{3} $(DATADIR)/stim1.csv 0.0 --mode=regular  --A_0=0.05e-9 --A_max=0.2e-9 --N=50 --rank={3} --d={1} --T=0.05 --f={2} --verbose --save_only' \
		::: -2e-3 ::: 20 ::: 1 2 3 48 49 50

exp16:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp16.log' \
		--nice 19 --delay 2 \
		'python budget_experiments.py ranked $(DATADIR)/exp16_d{1}_f{2}_r{3} $(DATADIR)/stim1.csv 0.0 --mode=regular  --A_0=0.05e-9 --A_max=0.2e-9 --N=50 --rank={3} --d={1} --T=0.033 --f={2} --verbose --save_only' \
		::: -2e-3 ::: 30 ::: 1 2 3 48 49 50

exp17:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp17.log' \
		--nice 19 --delay 2 \
		'python budget_experiments.py ranked $(DATADIR)/exp17_d{1}_f{2}_r{3} $(DATADIR)/stim1.csv 0.0 --mode=regular  --A_0=0.05e-9 --A_max=0.2e-9 --N=50 --rank={3} --d={1} --T=0.016 --f={2} --verbose --save_only' \
		::: -2e-3 ::: 60 ::: 1 2 3 48 49 50

# Explore multi-F using a small window T (eq to 60 Hz)
exp18:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp18.log' \
		--nice 19 --delay 2 \
		'python budget_experiments.py ranked $(DATADIR)/exp18_d{1}_f{2}_r{3} $(DATADIR)/stim1.csv 0.0 --mode=regular  --A_0=0.05e-9 --A_max=0.2e-9 --N=50 --rank={3} --d={1} --T=0.016 --f={2} --verbose --save_only' \
		::: -2e-3 ::: 4 8 12 20 30 60 ::: 1 3 48 50

# Explore multi-F using a big window T (eq to 4 Hz)
exp19:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp18.log' \
		--nice 19 --delay 2 \
		'python budget_experiments.py ranked $(DATADIR)/exp19_d{1}_f{2}_r{3} $(DATADIR)/stim1.csv 0.0 --mode=regular  --A_0=0.05e-9 --A_max=0.2e-9 --N=50 --rank={3} --d={1} --T=0.25 --f={2} --verbose --save_only' \
		::: -2e-3 ::: 4 8 12 20 30 60 ::: 1 3 48 50

# ------------------------------------------------------------------
# 1-26-2018
# More pareto - F search, with big and small T
# small T
exp20:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp20.log' \
		--nice 19 --delay 2 \
		'python budget_experiments.py pareto $(DATADIR)/exp20_d{1}_f{2} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_max=0.05e-9 --N=50 --d={1} --T=0.016 --f={2} --verbose' \
		::: -2e-3 ::: 4 8 12 20 30 60

# big T
exp21:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp21.log' \
		--nice 19 --delay 2 \
		'python budget_experiments.py pareto $(DATADIR)/exp21_d{1}_f{2} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_max=0.05e-9 --N=50 --d={1} --T=0.25 --f={2} --verbose' \
		::: -2e-3 ::: 4 8 12 20 30 60

# ------------------------------------------------------------------
# 1-27-2018
# Forward with small T, multi-f
exp22:
	parallel -j 10 -v \
		--joblog '$(DATADIR)/exp22.log' \
		--nice 19 --delay 2 \
		'python budget_experiments.py forward $(DATADIR)/exp22_d{1}_f{2} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=50 --d={1} --T=0.016 --f={2} --verbose --save_only' \
		::: -2e-3 ::: 4 8 12 20 24 30 40 60

# Forward with big T, multi-f
exp23:
	parallel -j 10 -v \
		--joblog '$(DATADIR)/exp23.log' \
		--nice 19 --delay 2 \
		'python budget_experiments.py forward $(DATADIR)/exp23_d{1}_f{2} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=50 --d={1} --T=0.25 --f={2} --verbose --save_only' \
		::: -2e-3 ::: 4 8 12 20 24 30 40 60

# ------------------------------------------------------------------
# 1-28-2017
# Forward: T and d and f, all matched, multi-f
# Use stim 2 (cont, sparse), 
exp24:
	parallel -j 10 -v \
		--joblog '$(DATADIR)/exp24.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py forward $(DATADIR)/exp24_d{1}_f{3} $(DATADIR)/stim2.csv 0.20 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=50 --d={1} --T={2} --f={3} --verbose --save_only --all_phi' \
		:::: exp24.txt 

# Forward: T and f, all matched, d=2 ms, multi-f
# Use stim 2 (cont, sparse), 
exp25:
	parallel -j 10 -v \
		--joblog '$(DATADIR)/exp25.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py forward $(DATADIR)/exp25_d{1}_f{3} $(DATADIR)/stim2.csv 0.20 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=50 --d={1} --T={2} --f={3} --verbose --save_only --all_phi' \
		:::: exp25.txt 

# rerun of 23, opt all phi
# Forward with big T, multi-f
exp26:
	parallel -j 10 -v \
		--joblog '$(DATADIR)/exp26.log' \
		--nice 19 --delay 2 \
		'python budget_experiments.py forward $(DATADIR)/exp26_d{1}_f{2} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=50 --d={1} --T=0.25 --f={2} --verbose --save_only --all_phi' \
		::: -2e-3 ::: 4 8 12 20 24 30 40 60

# -------------------------------------------------------------------
# 1-29-2018
# Created branch: lock_phi from consensus (39848f35cdbaf7d08dfc2d2ed89603ba9de960f9)
# forward, pareto, ranked changed so phi_w is opt nut all analyses are locked to phi_E
# and only A is run. 
# This is the more realistic case where the brain uses the possible peak power in a small
# window w, at a short delay d, to optimize the next cycle of oscillation BUT 
# the next cycle starts at E. 
# ...Perhaps this should have been the default approach all along...

# Pareto exp test
# Re-using exp25 config: d=-2e-3, T=1/f, f=(4,8,12,2024,30,40,60)
exp27:
	parallel -j 11 -v \
		--joblog '$(DATADIR)/exp27.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py pareto $(DATADIR)/exp27_d{1}_f{3} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_max=0.05e-9 --N=50 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp25.txt 

# Forward exp test
# Re-using exp25 config: d=-2e-3, T=1/f, f=(4,8,12,2024,30,40,60)
exp28:
	parallel -j 11 -v \
		--joblog '$(DATADIR)/exp28.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py forward $(DATADIR)/exp28_d{1}_f{3} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=50 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp25.txt 

# Rank exp test
# Explore multi-F using a big window T (eq to 4 Hz)
exp29:
	parallel -j 11 -v \
		--joblog '$(DATADIR)/exp29.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py ranked $(DATADIR)/exp29_d{1}_f{3}_r{4} $(DATADIR)/stim1.csv 0.0 --mode=regular  --A_0=0.05e-9 --A_max=0.2e-9 --N=50 --rank={4} --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp29.txt


# ---------------------------------------------------------------------
# 1-29-2018
# Re-run, using the network (group) for analysis not indiv neurons;
# Power are the cost of identifiability.

# Pareto exp test
# Re-using exp25 config: d=-2e-3, T=1/f, f=(4,8,12,2024,30,40,60)
exp30:
	parallel -j 11 -v \
		--joblog '$(DATADIR)/exp30.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py pareto $(DATADIR)/exp30_d{1}_f{3} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_max=0.05e-9 --N=50 --d={1} --T={2} --f={3} --verbose --save_only --score_group' \
		:::: exp25.txt 

# Forward exp test
# Re-using exp25 config: d=-2e-3, T=1/f, f=(4,8,12,2024,30,40,60)
exp31:
	parallel -j 11 -v \
		--joblog '$(DATADIR)/exp31.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py forward $(DATADIR)/exp31_d{1}_f{3} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=50 --d={1} --T={2} --f={3} --verbose --save_only --score_group' \
		:::: exp25.txt 

# ----------------------------------------------------------------------------
# 1-30-2018
# Removed w_in opt from Pareto; searching only A in this run 
# (compare to exp27/30)
# Re-using exp25 config: d=-2e-3, T=1/f, f=(4,8,12,2024,30,40,60)
exp32:
	parallel -j 11 -v \
		--joblog '$(DATADIR)/exp32.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py pareto $(DATADIR)/exp32_d{1}_f{3} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_max=0.05e-9 --N=50 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp25.txt 

# ----------------------------------------------------------------------------
# 2-3-2018
# Exp 28 is the choice for group exps. Rerun with much hugher N
exp33:
	parallel -j 11 -v \
		--joblog '$(DATADIR)/exp33.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py forward $(DATADIR)/exp33_d{1}_f{3} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=500 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp25.txt 

# ----------------------------------------------------------------------------
# 2-4-2018
# Need to be running with larger N. Prev results were misleading. (FUCKS!).
# When run en masse, it is clear V0 is too strong. Added a bias - A/2 ('homeostatic') correction
# to see what that does `--correct_bias`
exp34:
	parallel -j 11 -v \
		--joblog '$(DATADIR)/exp34.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py forward $(DATADIR)/exp34_d{1}_f{3} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=500 --d={1} --T={2} --f={3} --verbose --save_only --correct_bias' \
		:::: exp25.txt 

# Reverse - first attempt. Fit only A.
# KILLED after MANY days of RUNNING. 
exp35:
	parallel -j 2 -v \
		--joblog '$(DATADIR)/exp35.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py reverse $(DATADIR)/exp35_d{1}_f{3} $(DATADIR)/stim1.csv 0.0 --percent_change=0.1 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=10 --d={1} --T={2} --f_0={3} --verbose --save_only' \
		:::: exp25.txt 

# ----------------------------------------------------------------------------
# 2-12-2018
# Repeat of 34, this time reporting V_b to correct V_o/V_c as bias shifts
# The last run (34) was not difficult to interpret without this correction.
# To speed thing up I also decrease N a bit
exp36:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp36.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py forward $(DATADIR)/exp36_d{1}_f{3} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=300 --d={1} --T={2} --f={3} --verbose --save_only --correct_bias' \
		:::: exp25.txt 

# Exact repeat of 34, but I fixed a major bug in
# the code as it existed when 34 was run. Want to compare
# the two, hence this repeat.
exp37:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp37.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py forward $(DATADIR)/exp37_d{1}_f{3} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=300 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp25.txt 
# Result:
# - Useless. I forgot the --correct_bias flag. I ran the wrong exp.

# ----------------------------------------------------------------------------
# 2-13-2018
# Run perturbation experiments on rank 1, 300
# both correcting bias (38) and not (38)
exp38:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp38.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py perturb $(DATADIR)/exp38_d{1}_f{3}_r{4} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=300 --d={1} --T={2} --f={3} --rank={4} --max_percent=0.1 --n_perturbations=10 --verbose --save_only' \
		:::: exp38.txt 

exp39:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp39.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py perturb $(DATADIR)/exp39_d{1}_f{3}_r{4} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=300 --d={1} --T={2} --f={3} --rank={4} --max_percent=0.1 --n_perturbations=10 --verbose --save_only' \
		:::: exp38.txt 
# Result:
# - Bug in code. Ignore.

# Repeat of above, with n_perturbations = 100 (up from 10)
# q8b44d8c554b15c07a7be7dcca8f99b3862055598
exp40:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp40.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py perturb $(DATADIR)/exp40_d{1}_f{3}_r{4} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=300 --d={1} --T={2} --f={3} --rank={4} --max_percent=0.1 --n_perturbations=100 --verbose --save_only' \
		:::: exp38.txt 
# Result:
# - Bug in code. Ignore.

# ----------------------------------------------------------------------------
# 2-14-2018
# Perturb a much smaller range (0.02%) 
# 8b44d8c554b15c07a7be7dcca8f99b3862055598
exp41:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp41.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py perturb $(DATADIR)/exp41_d{1}_f{3}_r{4} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=300 --d={1} --T={2} --f={3} --rank={4} --max_percent=0.02 --n_perturbations=100 --verbose --save_only --correct_bias' \
		:::: exp38.txt 
# Result:
# - Bug in code. Ignore.

# ----------------------------------------------------------------------------
# 2-15-2018
# Repeat 41, but Analyzing neurons seperately
# c093a6dea0fd485f38ffe5a0f9a8e712c980b083
exp42:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp42.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py perturb $(DATADIR)/exp42_d{1}_f{3}_r{4} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=300 --d={1} --T={2} --f={3} --rank={4} --max_percent=0.02 --n_perturbations=100 --verbose --save_only --correct_bias' \
		:::: exp38.txt 
# Result:
# - Bug in code. Ignore.

# 2-22-2018
# wider perturb range...
exp43:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp43.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py perturb $(DATADIR)/exp43_d{1}_f{3}_r{4} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=300 --d={1} --T={2} --f={3} --rank={4} --max_percent=0.05 --n_perturbations=100 --verbose --save_only --correct_bias' \
		:::: exp38.txt
# Result:
# - Bug in code. Ignore. 

# EVEN wider perturb range...
exp44:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp44.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py perturb $(DATADIR)/exp44_d{1}_f{3}_r{4} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=300 --d={1} --T={2} --f={3} --rank={4} --max_percent=0.25 --n_perturbations=100 --verbose --save_only --correct_bias' \
		:::: exp38.txt 
# Result:
# - Bug in code. Ignore.

# -
# Revisit stim2, with and without homeostatsis.
# Extending A range
exp45:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp45.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py forward $(DATADIR)/exp45_d{1}_f{3} $(DATADIR)/stim2.csv 0.2 --mode=regular --A_0=0.05e-9 --A_max=0.3e-9 --N=300 --d={1} --T={2} --f={3} --verbose --save_only --correct_bias' \
		:::: exp25.txt 

exp46:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp46.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py forward $(DATADIR)/exp46_d{1}_f{3} $(DATADIR)/stim2.csv 0.2 --mode=regular --A_0=0.05e-9 --A_max=0.3e-9 --N=300 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp25.txt 
# Results:
# - Patterns consistent with stim1 exps. It's not the rise that cause positive var with A when there is no bias adjustment. 
# - I'm not sure how to rectify this prev. work on E osc and spike sync. 
# - Stim 2 analysis was done from .37 - .4 seconds. stim3 was created in part to put analysis in a mode sensible range,
#   and to make sure the two stim cases are different enough. 

# Stim 1, with wider A range
exp47:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp47.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py forward $(DATADIR)/exp47_d{1}_f{3} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_0=0.05e-9 --A_max=0.3e-9 --N=300 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp25.txt 
# Result:
# - I forgot the --correct_bias flag. Wrong exp was run.
 
# ----------------------------------------------------------------------------
# 2-23-2018
# Examine all the populations' A_hat effect single neuron 
exp48:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp48.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py population $(DATADIR)/exp48_d{1}_f{3}_r{4} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_0=0.05e-9 --A_max=0.3e-9 --N=100 --d={1} --T={2} --f={3} --verbose --save_only --correct_bias --analyze_rank={4}' \
		:::: exp48.txt 

# Examine all a single A_hat effect effect on the whole population
exp49:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp49.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py single $(DATADIR)/exp49_d{1}_f{3}_r{4} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_0=0.05e-9 --A_max=0.3e-9 --N=100 --d={1} --T={2} --f={3} --verbose --save_only --correct_bias --analyze_rank={4}' \
		:::: exp48.txt 	 

# Repeat 48/49 on stim2, without bias
# With bias correction
exp50:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp50.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py population $(DATADIR)/exp50_d{1}_f{3}_r{4} $(DATADIR)/stim2.csv 0.2 --mode=regular --A_0=0.05e-9 --A_max=0.3e-9 --N=100 --d={1} --T={2} --f={3} --verbose --save_only --analyze_rank={4}' \
		:::: exp48.txt 

exp51:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp51.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py single $(DATADIR)/exp51_d{1}_f{3}_r{4} $(DATADIR)/stim2.csv 0.2 --mode=regular --A_0=0.05e-9 --A_max=0.3e-9 --N=100 --d={1} --T={2} --f={3} --verbose --save_only --rank={4}' \
		:::: exp48.txt 

# ----------------------------------------------------------------------------
# 2-25-2018

# Single:
# Repeat 48/49 on *stim3*, without bias and
# with bias correction.
# scale w_in by 1/10 to account for high pop rate
# in stim 3
exp52:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp52.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py single $(DATADIR)/exp52_d{1}_f{3}_r{4} $(DATADIR)/stim3.csv 0.2 --mode=regular --A_0=0.05e-9 --A_max=0.3e-9 --N=100 --d={1} --T={2} --f={3} --verbose --save_only --rank={4} --scale_w_in=0.1 --correct_bias' \
		:::: exp48.txt 

exp53:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp53.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py single $(DATADIR)/exp53_d{1}_f{3}_r{4} $(DATADIR)/stim3.csv 0.2 --mode=regular --A_0=0.05e-9 --A_max=0.3e-9 --N=100 --d={1} --T={2} --f={3} --verbose --save_only --rank={4} --scale_w_in=0.1' \
		:::: exp48.txt 

# Population :
# Repeat 48/49 on *stim3*, without bias and
# with bias correction.
# scale w_in by 1/10 to account for high pop rate
# in stim 3
exp54:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp54.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py population $(DATADIR)/exp54_d{1}_f{3}_r{4} $(DATADIR)/stim3.csv 0.2 --mode=regular --A_0=0.05e-9 --A_max=0.3e-9 --N=100 --d={1} --T={2} --f={3} --verbose --save_only --analyze_rank={4} --scale_w_in=0.1 --correct_bias' \
		:::: exp48.txt 
# Stopped early on first run = up to f=24

# !
exp55:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp55.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py population $(DATADIR)/exp55_d{1}_f{3}_r{4} $(DATADIR)/stim3.csv 0.2 --mode=regular --A_0=0.05e-9 --A_max=0.3e-9 --N=100 --d={1} --T={2} --f={3} --verbose --save_only --analyze_rank={4} --scale_w_in=0.1' \
		:::: exp48.txt 

# Stim 3 - whole population analysis
# Corrected bias
# !
exp56:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp56.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py population $(DATADIR)/exp56_d{1}_f{3} $(DATADIR)/stim3.csv 0.2 --mode=regular --A_0=0.05e-9 --A_max=0.3e-9 --N=100 --d={1} --T={2} --f={3} --verbose --save_only --scale_w_in=0.1 --correct_bias' \
		:::: exp25.txt 

# Uncorrected bias
# !
exp57:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp57.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py population $(DATADIR)/exp57_d{1}_f{3} $(DATADIR)/stim3.csv 0.2 --mode=regular --A_0=0.05e-9 --A_max=0.3e-9 --N=100 --d={1} --T={2} --f={3} --verbose --save_only --scale_w_in=0.1' \
		:::: exp25.txt 

# ----------------------------------------------------------------------------
# 2-28-2016
# Re-think of exp plan. Re-think of aims. 
# branch `three_analyses` -> `re_aim`

# 5f537a1ee0a3c8826ce2ae66936c4d3c72bb7c22
# Sweep A wide and dense.
exp58:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp58.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py sweep_A $(DATADIR)/exp58_d{1}_f{3} $(DATADIR)/stim1.csv 0 --mode=regular --A_0=0.00e-9 --A_max=0.15e-9 --N=100 --n_samples=500 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp25.txt 

# min V_free, percent sweep 0.25 
exp59:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp59.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py min_max $(DATADIR)/exp59_d{1}_f{3} $(DATADIR)/stim1.csv 0 --mode=regular --A_0=0.00e-9 --A_max=0.2e-9 --target=min --N=100 --n_samples=100 --percent=0.25 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp25.txt 

# max V_free, percent sweep 0.25 
exp60:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp60.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py min_max $(DATADIR)/exp60_d{1}_f{3} $(DATADIR)/stim1.csv 0 --mode=regular --A_0=0.00e-9 --A_max=0.2e-9 --target=max --N=100 --n_samples=100 --percent=0.25 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp25.txt 

# min V_free, percent sweep 200%
exp61:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp61.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py min_max $(DATADIR)/exp61_d{1}_f{3} $(DATADIR)/stim1.csv 0 --mode=regular --A_0=0.00e-9 --A_max=0.2e-9 --target=min --N=100 --n_samples=100 --percent=2 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp25.txt 

# Search delays, with min V_free
exp62:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp62.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py min_max $(DATADIR)/exp62_d{1}_f{3} $(DATADIR)/stim1.csv 0 --mode=regular --A_0=0.00e-9 --A_max=0.2e-9 --target=min --N=100 --n_samples=100 --percent=0.5 --d={1} --T={2} --f={3} --verbose --save_only' \
		::: -2e-3 -4e-3 -6e-3 -8e-3 -10e-3 ::: 0.05 ::: 20 


# min V_free, percent sweep 100%, with d=4
exp63:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp63.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py min_max $(DATADIR)/exp63_d{1}_f{3} $(DATADIR)/stim1.csv 0 --mode=regular --A_0=0.00e-9 --A_max=0.2e-9 --target=min --N=100 --n_samples=100 --percent=1 --d={1} --T={2} --f={3} --verbose --save_only' \
		::: -4e-3 ::: 0.05 ::: 20 

# -----------------------------------------------
# 3-2-2018

# repeat: min V_free, percent sweep 100%, with d=4
# increaes n_samples
# is the jitter on err/var due to spike number changes?
exp64:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp64.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py min_max $(DATADIR)/exp64_d{1}_f{3} $(DATADIR)/stim1.csv 0 --mode=regular --A_0=0.00e-9 --A_max=0.2e-9 --target=min --N=100 --n_samples=2000 --percent=1 --d={1} --T={2} --f={3} --verbose --save_only' \
		::: -4e-3 ::: 0.05 ::: 20 

# sweep, with higher density
# is the jitter on err/var due to spike number changes?
exp65:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp65.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py sweep_A $(DATADIR)/exp65_d{1}_f{3} $(DATADIR)/stim1.csv 0 --mode=regular --A_0=0.00e-9 --A_max=0.10e-9 --N=100 --n_samples=2000 --d={1} --T={2} --f={3} --verbose --save_only' \
		::: -4e-3 ::: 0.05 ::: 20 

# No lock exp:
# Search delays, with min V_free
# Es were randomly generated.
exp66:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp62.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py min_max $(DATADIR)/exp62_d{1}_f{3}_E{4} $(DATADIR)/stim1.csv {4} --mode=regular --A_0=0.00e-9 --A_max=0.2e-9 --target=min --N=100 --n_samples=20 --percent=0.5 --d={1} --T={2} --f={3} --verbose --save_only --no_lock' \
		::: -2e-3 -4e-3 -6e-3 ::: 0.05 ::: 20 ::: 0.2368905 0.23911531 0.22072062 0.22945046 0.22405547

# -----------------------------------------------
# 3-6-2018
# sweep, low n_samples
exp67:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp67.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py sweep_A $(DATADIR)/exp67_d{1}_f{3} $(DATADIR)/stim1.csv 0 --mode=regular --A_0=0.00e-9 --A_max=0.15e-9 --N=100 --n_samples=20 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp64.txt

# very high n_samples
exp68:
	parallel -j 8 -v \
		--joblog '$(DATADIR)/exp68.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py sweep_A $(DATADIR)/exp68_d{1}_f{3} $(DATADIR)/stim1.csv 0 --mode=regular --A_0=0.00e-9 --A_max=0.15e-9 --N=100 --n_samples=500 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp64.txt


# no lock, several E,d combinations
exp69:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp69.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py min_max $(DATADIR)/exp69_d{1}_f{3}_E{4} $(DATADIR)/stim1.csv {4} --mode=regular --A_0=0.00e-9 --A_max=0.2e-9 --target=min --N=100 --n_samples=20 --percent=0.5 --d={1} --T={2} --f={3} --verbose --save_only --no_lock' \
		::: -2e-3 -4e-3 -6e-3 ::: 0.05 ::: 20 ::: 0.2368905 0.23911531 0.22072062 0.22945046 0.22405547

# -----------------------------------------------
# 3-7-2019
# 9f79cf8a4ae8deb2bb9981e2355c02e6a1f1dfd5
# after above commit sweep_A saves all n results.
# Instead of just the mean. 

# Re-running 68
# very high n_samples
exp70:
	parallel -j 8 -v \
		--joblog '$(DATADIR)/exp70.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py sweep_A $(DATADIR)/exp70_d{1}_f{3} $(DATADIR)/stim1.csv 0 --mode=regular --A_0=0.00e-9 --A_max=0.20e-9 --N=260 --n_samples=1000 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp64.txt

# -----------------------------------------------
# 3-8-2019
# Re-running 70 - oscillations
# now happend as 1-cycle bursts
# 459ae7c5f60a9b04c987553540fb73c05d989f3c
# Re-running 70 to see if this change prevents
# unexplained steps in var/error 
# (Not hopeful but this is a good change anyway)
exp71:
	parallel -j 8 -v \
		--joblog '$(DATADIR)/exp71.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py sweep_A $(DATADIR)/exp71_d{1}_f{3} $(DATADIR)/stim1.csv 0 --mode=regular --A_0=0.00e-9 --A_max=0.20e-9 --N=260 --n_samples=1000 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp64.txt

# -----------------------------------------------
# 3-9-2018
# fixed bug in w targeting 
# 5aadf1d487f66960c16a738cffed74ca16a66742
exp72:
	parallel -j 8 -v \
		--joblog '$(DATADIR)/exp72.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py sweep_A $(DATADIR)/exp72_d{1}_f{3} $(DATADIR)/stim1.csv 0 --mode=regular --A_0=0.00e-9 --A_max=0.20e-9 --N=250 --n_samples=1000 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp64.txt


# exp71/2 wih Z=(A*0.5) homeostatic correction,
# also samped fewer f bins for speed
exp73:
	parallel -j 4 -v \
		--joblog '$(DATADIR)/exp73.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py sweep_A $(DATADIR)/exp73_d{1}_f{3} $(DATADIR)/stim1.csv 0 --mode=regular --A_0=0.00e-9 --A_max=0.20e-9 --Z=0.5 --N=250 --n_samples=1000 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp72.txt

# -----------------------------------------------
# 3-10-2018
# Explore a range of Z
exp74:
	parallel -j 12 -v \
		--joblog '$(DATADIR)/exp74.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py sweep_A $(DATADIR)/exp74_d{1}_f{3}_Z{4} $(DATADIR)/stim1.csv 0 --mode=regular --A_0=0.00e-9 --A_max=0.20e-9 --Z={4} --N=250 --n_samples=1000 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp72.txt ::: 0.1 0.2 0.3 

# -----------------------------------------------
# 3-11-2018
# Explore more slow freqs
exp75:
	parallel -j 9 -v \
		--joblog '$(DATADIR)/exp75.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py sweep_A $(DATADIR)/exp75_d{1}_f{3} $(DATADIR)/stim1.csv 0 --mode=regular --A_0=0.00e-9 --A_max=0.20e-9 --N=250 --n_samples=1000 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp75.txt

# -----------------------------------------------
# 3-12-2018
# MAJOR BUGFIX in low/high freq est of V.B.
# 191bc8a6face2e0671b606fd94b71a6abb168d07
#
# Homeostatis exp set 1 - try a range of As and freqs
# Careful with `parallel`ism here - PSO also parallel
exp76:
	parallel -j 1 -v \
		--joblog '$(DATADIR)/exp76.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py homeostasis $(DATADIR)/exp76_d{1}_f{3}_A{4} $(DATADIR)/stim1.csv 0 --mode=regular --n_jobs=5 --A={4} --N=250 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp72.txt ::: 0.01e-9 0.025e-9 0.05e-9 0.075e-9 0.1e-9

# Re-run of exp72 after major bugfix in low/high freq in V.B. estimation.
exp77:
	parallel -j 8 -v \
		--joblog '$(DATADIR)/exp77.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py sweep_A $(DATADIR)/exp77_d{1}_f{3} $(DATADIR)/stim1.csv 0 --mode=regular --A_0=0.00e-9 --A_max=0.20e-9 --N=250 --n_samples=1000 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp64.txt

# Explore a range of Z
exp78:
	parallel -j 12 -v \
		--joblog '$(DATADIR)/exp78.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py sweep_A $(DATADIR)/exp78_d{1}_f{3}_Z{4} $(DATADIR)/stim1.csv 0 --mode=regular --A_0=0.00e-9 --A_max=0.20e-9 --Z={4} --N=250 --n_samples=1000 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp72.txt ::: 0.1 0.2 0.3 

# 3-13-2018
# Explore more slow freq (re-run of 75)
exp79:
	parallel -j 9 -v \
		--joblog '$(DATADIR)/exp79.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py sweep_A $(DATADIR)/exp79_d{1}_f{3} $(DATADIR)/stim1.csv 0 --mode=regular --A_0=0.00e-9 --A_max=0.20e-9 --N=250 --n_samples=1000 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp75.txt

# Explore more fast freqs 
exp80:
	parallel -j 9 -v \
		--joblog '$(DATADIR)/exp80.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py sweep_A $(DATADIR)/exp80_d{1}_f{3} $(DATADIR)/stim1.csv 0 --mode=regular --A_0=0.00e-9 --A_max=0.20e-9 --N=250 --n_samples=1000 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp80.txt

# 3-16-2018
# fixed bug in 76/homeostatis. re-running.
exp81:
	parallel -j 1 -v \
		--joblog '$(DATADIR)/exp81.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py homeostasis $(DATADIR)/exp81_d{1}_f{3}_A{4} $(DATADIR)/stim1.csv 0 --mode=regular --n_jobs=5 --A={4} --N=250 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp72.txt ::: 0.01e-9 0.025e-9 0.05e-9 0.075e-9 0.1e-9

# 3-20-2018
# Run several 'optimal' experiments,
exp82_85:
	python budget_experiments.py optimal $(DATADIR)/exp82_max $(DATADIR)/stim1.csv  0.0 0.1 --n_samples=100 --alg=max_deviants --save_only --verbose
	python budget_experiments.py optimal $(DATADIR)/exp83_left $(DATADIR)/stim1.csv  0.0 0.1 --n_samples=100 --alg=left_deviants --save_only --verbose
	python budget_experiments.py optimal $(DATADIR)/exp84_uni $(DATADIR)/stim1.csv  0.0 0.1 --n_samples=100 --alg=uniform --save_only --verbose
	python budget_experiments.py optimal $(DATADIR)/exp85_cc $(DATADIR)/stim1.csv  0.0 0.1 --n_samples=100 --alg=coincidence --save_only --verbose

# Save some example spikes
exp86_89:
	python budget_experiments.py optimal $(DATADIR)/exp86_max $(DATADIR)/stim1.csv  0.05 0.05 --n_samples=1 --alg=max_deviants --save_only --verbose --save_spikes
	python budget_experiments.py optimal $(DATADIR)/exp87_left $(DATADIR)/stim1.csv  0.05 0.05 --n_samples=1 --alg=left_deviants --save_only --verbose --save_spikes
	python budget_experiments.py optimal $(DATADIR)/exp88_uni $(DATADIR)/stim1.csv  0.05 0.05 --n_samples=1 --alg=uniform --save_only --verbose --save_spikes
	python budget_experiments.py optimal $(DATADIR)/exp89_cc $(DATADIR)/stim1.csv  0.05 0.05 --n_samples=1 --alg=coincidence --save_only --verbose --save_spikes

# -----------------------------------------------
# 3-29-2017
# Robustness to noise, at 20 Hz
exp90:
	parallel -j 3 -v \
		--joblog '$(DATADIR)/exp90.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py sweep_A $(DATADIR)/exp90_sigma{1} $(DATADIR)/stim1.csv 0 --mode=regular --A_0=0.00e-9 --A_max=0.15e-9 --N=250 --n_samples=300 --d=-2e-3 --T=0.05 --f=20 --verbose --save_only --sigma={1}' \
		::: 1.0e-9 5.0e-9 10.0e-9

# -----------------------------------------------
# 3-30-2017
# Found a minor bug in adex math, checking to make sure there are no difference
# 6bf421f0e93c792331bfaa24eb48671b1867f748 

# Re-run of exp72 
exp91:
	parallel -j 8 -v \
		--joblog '$(DATADIR)/exp91.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py sweep_A $(DATADIR)/exp91_d{1}_f{3} $(DATADIR)/stim1.csv 0 --mode=regular --A_0=0.00e-9 --A_max=0.15e-9 --N=250 --n_samples=300 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp64.txt

# rerun of 78
# Explore a range of Z
exp92:
	parallel -j 3 -v \
		--joblog '$(DATADIR)/exp92.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py sweep_A $(DATADIR)/exp92_Z{1} $(DATADIR)/stim1.csv 0 --mode=regular --A_0=0.00e-9 --A_max=0.15e-9 --N=250 --n_samples=300 --d=-2e-3 --T=0.05 --f=20 --verbose --save_only --Z={1}' \
		::: 0.1 0.2 0.3

# -----------------------------------------------
# 3-30-2017
# Running first set of exps with a 'hetero' population
exp93:
	parallel -j 8 -v \
		--joblog '$(DATADIR)/exp93.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py sweep_A $(DATADIR)/exp93_d{1}_f{3} $(DATADIR)/stim1.csv 0 --mode=heterogenous --A_0=0.00e-9 --A_max=0.15e-9 --N=250 --n_samples=300 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp64.txt

# 4-2-2018
# mad for ith neurons now refers to the pop mean...
# c5110d5ae92dd7f0511c00f5c7ea4266863df397 

# More re-run - this time of fast/flow f
# Explore more slow freq (re-run of 75)
exp94:
	parallel -j 9 -v \
		--joblog '$(DATADIR)/exp94.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py sweep_A $(DATADIR)/exp94_d{1}_f{3} $(DATADIR)/stim1.csv 0 --mode=regular --A_0=0.00e-9 --A_max=0.15e-9 --N=250 --n_samples=150 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp75.txt

# Explore more fast freqs 
exp95:
	parallel -j 9 -v \
		--joblog '$(DATADIR)/exp95.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py sweep_A $(DATADIR)/exp95_d{1}_f{3} $(DATADIR)/stim1.csv 0 --mode=regular --A_0=0.00e-9 --A_max=0.15e-9 --N=250 --n_samples=150 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp80.txt

# (Re-run of exp72)
exp96:
	parallel -j 8 -v \
		--joblog '$(DATADIR)/exp96.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py sweep_A $(DATADIR)/exp96_d{1}_f{3} $(DATADIR)/stim1.csv 0 --mode=regular --A_0=0.00e-9 --A_max=0.15e-9 --N=250 --n_samples=150 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp64.txt

# Rerun noise
exp97:
	parallel -j 3 -v \
		--joblog '$(DATADIR)/exp97.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py sweep_A $(DATADIR)/exp97_sigma{1} $(DATADIR)/stim1.csv 0 --mode=regular --A_0=0.00e-9 --A_max=0.15e-9 --N=250 --n_samples=150 --d=-2e-3 --T=0.05 --f=20 --verbose --save_only --sigma={1}' \
		::: 1.0e-9 2.0e-9 3.0e-9 4.0e-9 5.0e-9 

# Rerun Z
exp98:
	parallel -j 3 -v \
		--joblog '$(DATADIR)/exp98.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py sweep_A $(DATADIR)/exp98_Z{1} $(DATADIR)/stim1.csv 0 --mode=regular --A_0=0.00e-9 --A_max=0.15e-9 --N=250 --n_samples=150 --d=-2e-3 --T=0.05 --f=20 --verbose --save_only --Z={1}' \
		::: 0.1 0.2 0.3
	
# Explore strong stim4 (from 1000 cells)
# with much reduced w_in
# (Re-run of exp72)
exp99:
	parallel -j 3 -v \
		--joblog '$(DATADIR)/exp99.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py sweep_A $(DATADIR)/exp99_d{1}_f{3} $(DATADIR)/stim4.csv 0 --mode=regular --w_in_0=3e-10 --w_in_max=3e-09 --A_0=0.00e-9 --A_max=0.15e-9 --N=250 --n_samples=150 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp64.txt

# Explore strong stim5 (from 10 cells)
# with much increased w_in
# (Re-run of exp72)
exp100:
	parallel -j 8 -v \
		--joblog '$(DATADIR)/exp100.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py sweep_A $(DATADIR)/exp100_d{1}_f{3} $(DATADIR)/stim5.csv 0 --mode=regular --w_in_0=4.5e-09 --w_in_max=45e-09 --A_0=0.00e-9 --A_max=0.15e-9 --N=250 --n_samples=150 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp64.txt

# rerun hetero exps ...
# Due to spontanouse bursts het exps need to give the model a hint for E,
# using 0.2 here (stim onset time).
exp101:
	parallel -j 8 -v \
		--joblog '$(DATADIR)/exp101.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py sweep_A $(DATADIR)/exp101_d{1}_f{3} $(DATADIR)/stim1.csv 0.2 --mode=heterogenous --A_0=0.00e-9 --A_max=0.15e-9 --N=250 --n_samples=150 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp64.txt

# -----------------------------------------------
# Fix the runtime to be long, and est sync error over it for many cycles
# consider 4 f: 10, 20, 10 Hz
exp102:
	parallel -j 6 -v \
		--joblog '$(DATADIR)/exp102.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py sweep_A $(DATADIR)/exp102_f{1}_t{2} $(DATADIR)/stim6.csv 0.2 --mode=regular --t={2} --t_pad=0.1 --A_0=0.00e-9 --A_max=0.1e-9 --N=50 --n_samples=20 --d=-2e-3 --T={2} --f={1} --n_cycles=100 --verbose --save_only' \
		::: 10 20 30 ::: 0.25 0.5 1 1.5 2 2.5 3.0


# ---------------------------------------------
# 3/26/2019
# Re-run of exp72 but using opt_sweep.
exp103:
	parallel -j 4 -v \
		--joblog '$(DATADIR)/exp103.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py optimal_sweep_A $(DATADIR)/exp103_d{1}_f{3} $(DATADIR)/stim1.csv 0 --mode=regular --A_0=0.00e-9 --A_max=0.02e-9 --N=250 --n_samples=10 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp64.txt


# 103 looks great. re-running with more samples
# Re-run of exp72 but using opt_sweep.
exp104:
	parallel -j 12 -v \
		--joblog '$(DATADIR)/exp104.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py optimal_sweep_A $(DATADIR)/exp104_d{1}_f{3} $(DATADIR)/stim1.csv 0 --mode=regular --A_0=0.00e-9 --A_max=0.02e-9 --N=250 --n_samples=150 --d={1} --T={2} --f={3} --verbose --save_only' \
		:::: exp64.txt