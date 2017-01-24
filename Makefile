lif_10000:
	-rm data/lif1_10000.csv
	python exp/opt1.py data/lif1_10000 10000 -a 30e-3 --lif 

adex_10000:
	-rm data/adex1_10000.csv
	python exp/opt1.py data/adex1_10000 10000 -a 10e-10 --adex

lif2_10000:
	-rm data/lif2_10000.csv
	python exp/opt2.py data/lif2_10000 10000 -a 30e-3 -w 0.6e-9  --lif

adex2_10000:
	-rm data/adex2_10000.csv
	python exp/opt2.py data/adex2_10000 10000 -a 30e-3 -w 0.6e-9  --adex