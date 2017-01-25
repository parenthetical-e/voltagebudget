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
	python exp/opt2.py data/adex2_10000 10000 -a 30e-3 -w 0.6e-9  --adex

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

# Make sigma_in a parameter, and run
# opt 5:
# max C, max sigma_Y, 
lif5_10000:
	-rm data/lif5_10000.csv
	python exp/opt5.py data/lif5_10000 10000 -a 30e-3 --lif 

adex5_10000:
	-rm data/adex5_10000.csv
	python exp/opt5.py data/adex5_10000 10000 -a 10e-10 --adex

# opt 6:
# max C, max sigma_Y, max A
# max C, max sigma_Y, 
lif6_10000:
	-rm data/lif6_10000.csv
	python exp/opt6.py data/lif6_10000 10000 -a 30e-3 --lif 

adex6_10000:
	-rm data/adex6_10000.csv
	python exp/opt6.py data/adex6_10000 10000 -a 10e-10 --adex
