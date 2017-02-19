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
opt20:
	-rm data/opt20_*
	parallel -j 6 -v \
		--joblog 'data/log' \
		--nice 19 \
		'python exp/opt20.py data/opt20_f{1} 10000 -a 5e-3 -w 0.3e-9 -t 0.1 -f {1} --lif' ::: \
			8 12 20 40

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
		'python exp/opt21.py data/opt23_f{1} 1000 -a 2e-10 -w 0.3e-9 -t 0.1 -f {1} --adex' ::: \
			8 12 20 40 

# ----------------------------------------------------------------
amp1:
	-rm data/amp1.csv 
	python exp/amp.py data/amp1 -n 10 -f 50 -a 50e-3 --lif

# Increase `w_in` a bit
amp2:
	-rm data/amp2.csv 
	python exp/amp.py data/amp2 -n 10 -w 0.3e-9 -f 50 -a 50e-3 --lif

# Restriced the A range; large values were leading to double (non stim)
# spikes
# Increase `w_in` a bit
amp3:
	-rm data/amp3.csv 
	python exp/amp.py data/amp3 -n 10 -w 0.3e-9 -f 50 -a 5e-3 --lif

amp4:
	-rm data/amp4.csv 
	python exp/amp.py data/amp4 -n 10 -w 0.6e-9 -f 50 -a 5e-10 --adex

# Denser sampling.
amp5:
	-rm data/amp5.csv 
	python exp/amp.py data/amp5 -n 10 -w 0.2e-9 -f 50 -a 20e-3 --lif --n_grid 50

amp6:
	-rm data/amp6.csv 
	python exp/amp.py data/amp6 -n 10 -w 0.6e-9 -f 50 -a 10e-10 --adex --n_grid 50

# Explore freq and t_stim over a dense range of A
amp10:
	-rm data/amp10*csv
	parallel -j 6 -v \
		--joblog 'data/log' \
		--nice 19 \
		'python exp/amp.py data/amp10_f{1}_t{2} -n 20 -w 0.3e-9 -f {1} -t {2} -a 15e-10 --adex --n_grid 75' ::: \
			8 12 20 40 ::: \
			0.1 0.12 0.14 0.15 0.16 0.18 

# lif of 10
amp11:
	-rm data/amp11*csv
	parallel -j 6 -v \
		--joblog 'data/log' \
		--nice 19 \
		'python exp/amp.py data/amp11_f{1}_t{2} -n 20 -w 0.3e-9 -f {1} -t {2} -a 20e-3 --lif --n_grid 75' ::: \
			8 12 20 40 ::: \
			0.1 0.12 0.14 0.15 0.16 0.18 
