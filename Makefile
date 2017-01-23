lif_10000:
	-rm data/lif_10000.csv
	python exp/opt.py data/lif_10000 10000 -a 30e-3 --lif 

adex_10000:
	-rm data/adex_10000.csv
	python exp/opt.py data/adex_10000 10000 -a 10e-10 --adex

opt2_test:
	python exp/opt2.py data/test2 10 -a 30e-3 -w 0.6e-9 -b 10e-3  --lif