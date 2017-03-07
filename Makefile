# ---------------------------------------------------------------------
# Spiking oscillations
# ---------------------------------------------------------------------

lif1_10000:
	-rm data/lif1_10000.csv
	python exp/opt1.py data/lif1_10000 10000 -a 30e-3 --lif 

adex1_10000:
	-rm data/adex1_10000.csv
	python exp/opt1.py data/adex1_10000 10000 -a 10e-10 --adex

lif2_10000:
	-rm data/lif2_10000.csv
	python exp/opt2.py data/lif2_10000 10000 -a 30e-3 -w 0.6e-9  --lif

adex2_10000:
	-rm data/adex2_10000.csv
	python exp/opt2.py data/adex2_10000 10000 -a 10e-10 -w 0.6e-9  --adex

lif3_10000:
	-rm data/lif3_10000.csv
	python exp/opt3.py data/lif3_10000 10000 -w 0.9e-9  --lif

adex3_10000:
	-rm data/adex3_10000.csv
	python exp/opt3.py data/adex3_10000 10000 -w 0.9e-9  --adex

# Min A, min sigma_Y
lif4_10000:
	-rm data/lif4_10000.csv
	python exp/opt4.py data/lif4_10000 10000 -a 30e-3 --lif 

adex4_10000:
	-rm data/adex4_10000.csv
	python exp/opt4.py data/adex4_10000 10000 -a 10e-10 --adex

# ---------------------------------------------------------------------
# Subthreshold oscillations
# ---------------------------------------------------------------------

# Make sigma_in a parameter, and run
# opt 5:
# max C, max sigma_Y, 
lif5_10000:
	-rm data/lif5_10000.csv
	python exp/opt5.py data/lif5_10000 10000 -a 5e-3 --lif 

adex5_10000:
	-rm data/adex5_10000.csv
	python exp/opt5.py data/adex5_10000 10000 -a 5e-10 --adex

# opt 6:
# max C, max sigma_Y, max A
lif6_10000:
	-rm data/lif6_10000.csv
	python exp/opt6.py data/lif6_10000 10000 -a 5e-3 --lif 

adex6_10000:
	-rm data/adex6_10000.csv
	python exp/opt6.py data/adex6_10000 10000 -a 5e-10 --adex


# opt 7:
# max C, max sigma_comp
lif7_10000:
	-rm data/lif7_10000.csv
	python exp/opt7.py data/lif7_10000 10000 -a 5e-3 --lif 

adex7_10000:
	-rm data/adex7_10000.csv
	python exp/opt7.py data/adex7_10000 10000 -a 5e-10 --adex

# ----------------------------------------------------------------
# Rerun -C, A opt but for subthreshold oscillations only

# max C, max Vfree
lif8_10000:
	-rm data/lif8_10000.csv
	python exp/opt8.py data/lif8_10000 10000 -a 5e-3 --lif 

adex8_10000:
	-rm data/adex8_10000.csv
	python exp/opt8.py data/adex8_10000 10000 -a 5e-10 --adex

# max C, max Vfree, larger A range compared to '7'
exp11: lif11_10000 adex10_10000 

lif11_10000:
	-rm data/lif11_10000.csv
	python exp/opt8.py data/lif11_10000 10000 -a 30e-3 --lif 

adex11_10000:
	-rm data/adex11_10000.csv
	python exp/opt8.py data/adex11_10000 10000 -a 10e-10 --adex

# max C, min Vfree
exp12: lif12_10000 adex12_10000

lif12_10000:
	-rm data/lif12_10000.csv
	python exp/opt9.py data/lif12_10000 10000 -a 5e-3 --lif 

adex12_10000:
	-rm data/adex12_10000.csv
	python exp/opt9.py data/adex12_10000 10000 -a 5e-10 --adex

# max C, min Vfree, larger A range compared to '12'
exp13: lif13_10000 adex13_10000

lif13_10000:
	-rm data/lif13_10000.csv
	python exp/opt9.py data/lif13_10000 10000 -a 30e-3 --lif 

adex13_10000:
	-rm data/adex13_10000.csv
	python exp/opt9.py data/adex13_10000 10000 -a 10e-10 --adex


# --
lif9_10000:
	-rm data/lif9_10000.csv
	python exp/opt1.py data/lif9_10000 10000 -a 5e-3 --lif 

adex9_10000:
	-rm data/adex9_10000.csv
	python exp/opt1.py data/adex9_10000 10000 -a 5e-10 --adex

lif10_10000:
	-rm data/lif10_10000.csv
	python exp/opt2.py data/lif10_10000 10000 -a 5e-3 -w 0.6e-9  --lif

adex10_10000:
	-rm data/adex10_10000.csv
	python exp/opt2.py data/adex10_10000 10000 -a 5e-10 -w 0.6e-9  --adex

# ----------------------------------------------------------------
# Search both A, and comp
# Opt for C, and sigma_y 

# For lif comp is sigma_in
# A sigma phi
opt20a:
	-rm data/opt20a_*
	python exp/opt20a.py data/opt20a_f8 10000 -a 3e-3 -w 0.3e-9 -t 0.14 -f 8 --lif
	python exp/opt20a.py data/opt20a_f20 10000 -a 3e-3 -w 0.3e-9 -t 0.11 -f 20 --lif
	python exp/opt20a.py data/opt20a_f40 10000 -a 3e-3 -w 0.3e-9 -t 0.105 -f 40 --lif

# A sigma
opt20b:
	-rm data/opt20b_*
	python exp/opt20b.py data/opt20b_f8 10000 -a 3e-3 -w 0.3e-9 -t 0.14 -f 8 --lif
	python exp/opt20b.py data/opt20b_f20 10000 -a 3e-3 -w 0.3e-9 -t 0.11 -f 20 --lif
	python exp/opt20b.py data/opt20b_f40 10000 -a 3e-3 -w 0.3e-9 -t 0.105 -f 40 --lif

# phi sigma
opt20c:
	-rm data/opt20c_*
	python exp/opt20c.py data/opt20c_f8 10000 -a 3e-3 -w 0.3e-9 -t 0.14 -f 8 --lif
	python exp/opt20c.py data/opt20c_f20 10000 -a 3e-3 -w 0.3e-9 -t 0.11 -f 20 --lif
	python exp/opt20c.py data/opt20c_f40 10000 -a 3e-3 -w 0.3e-9 -t 0.105 -f 40 --lif


# For adex comp is {a, b, Ereset}
opt21:
	-rm data/opt21_*
	parallel -j 6 -v \
		--joblog 'data/log' \
		--nice 19 \
		'python exp/opt21.py data/opt21_f{1} 10000 -a 2e-10 -w 0.3e-9 -t 0.1 -f {1} --adex' ::: \
			8 12 20 40 


# For lif comp is sigma_in
# A opt individually for each ith neuron
opt22:
	-rm data/opt22_*
	parallel -j 6 -v \
		--joblog 'data/log' \
		--nice 19 \
		'python exp/opt22.py data/opt22_f{1} 1000 -a 5e-3 -w 0.3e-9 -t 0.1 -f {1} --lif' ::: \
			8 12 20 40

# For adex comp is {a, b, Ereset}
# A opt individually for each ith neuron
opt23:
	-rm data/opt23_*
	parallel -j 6 -v \
		--joblog 'data/log' \
		--nice 19 \
		'python exp/opt23.py data/opt23_f{1} 1000 -a 2e-10 -w 0.3e-9 -t 0.1 -f {1} --adex' ::: \
			8 12 20 40 


# ----------------------------------------------------------------
# Opt using only budget terms
#
# For lif comp is sigma_in
# A sigma phi
opt30a:
	-rm data/opt30a_*
	python exp/opt30a.py data/opt30a_f8 10000 -a 3e-3 -w 0.15e-9 -t 0.14 -f 8 --lif


# For adex comp is sigma_in
# A sigma phi
opt31a:
	-rm data/opt31a_*
	python exp/opt31a.py data/opt31a_f8 10000 -a .5e-10 -w 0.15e-9 -t 0.14 -f 8 --adex


# ----------------------------------------------------------------
# Explore freq and t_stim over a dense range of A
# -t set to place spikes near or at oscillation peak
# lif
amp1:
	-rm data/amp1*csv
	python exp/amp.py data/amp1_f8 -n 100 -w 0.15e-9 -f 8 -t 0.14 -a 4e-3 --lif --n_grid 50
	python exp/amp.py data/amp1_f20 -n 100 -w 0.15e-9 -f 20 -t 0.11 -a 4e-3 --lif --n_grid 50
	python exp/amp.py data/amp1_f40 -n 100 -w 0.15e-9 -f 40 -t 0.11 -a 4e-3 --lif --n_grid 60

# adex
amp2:
	-rm data/amp2*csv
	python exp/amp.py data/amp2_f8 -n 100 -w 0.15e-9 -f 8 -t 0.14 -a 0.5e-10 --adex --n_grid 50
	python exp/amp.py data/amp2_f20 -n 100 -w 0.15e-9 -f 20 -t 0.11 -a 0.5e-10 --adex --n_grid 50
	python exp/amp.py data/amp2_f40 -n 100 -w 0.15e-9 -f 40 -t 0.11 -a 0.5e-10 --adex --n_grid 50
