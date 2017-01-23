lif_10000:
	-rm data/lif_10000.csv
	python exp/opt.py data/lif_10000 10000 -a 30e-3 --lif 

adex_10000:
	-rm data/adex_10000.csv
	python exp/opt.py data/adex_10000 10000 -a 10e-10 --adex
