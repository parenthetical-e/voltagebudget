lif_10000:
	-rm data/lif_10000.csv
	python exp/opt.py data/lif_10000 10000 -a 30e-3 --lif 

adex_10000:
	-rm data/adex_10000.csv
	python exp/opt.py data/adex_10000 10000 -a 10e-10 --adex

lif2_10000:
	-rm data/lif2_10000.csv
	python exp/opt2.py data/lif2_10000 10000 -a 30e-3 -w 0.6e-9 -b 10e-3  --lif