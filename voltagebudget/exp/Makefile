SHELL=/bin/bash -O expand_aliases
DATADIR=/home/ejp/src/voltagebudget/voltagebudget/data
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
exp35:
	parallel -j 2 -v \
		--joblog '$(DATADIR)/exp35.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py reverse $(DATADIR)/exp35_d{1}_f{3} $(DATADIR)/stim1.csv 0.0 --percent_change=0.1 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=10 --d={1} --T={2} --f_0={3} --verbose --save_only' \
		:::: exp25.txt 

# Repeat of 34, this time reporting V_b to correct V_o/V_c as bias shifts
# The last run (34) was not difficult to interpret without this correction.
# To speed thing up I also decrease N a bit
exp36:
	parallel -j 11 -v \
		--joblog '$(DATADIR)/exp36.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python budget_experiments.py forward $(DATADIR)/exp36_d{1}_f{3} $(DATADIR)/stim1.csv 0.0 --mode=regular --A_0=0.05e-9 --A_max=0.2e-9 --N=300 --d={1} --T={2} --f={3} --verbose --save_only --correct_bias' \
		:::: exp25.txt 
