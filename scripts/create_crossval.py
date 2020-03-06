"""

Create 10-fold cross-validation data from LitBank data.

"""

import sys, os, re

def create_data(ids, infolder, outfile):
	out=open(outfile, "w", encoding="utf-8")

	for idd in ids:
		infile="%s/%s.conll" % (infolder, idd)

		with open(infile) as file:
			for line in file:
				out.write("%s\n" % line.rstrip())

	out.close()

def get_ids_from_filename(filename):
	ids=[]
	with open(filename) as file:
		for line in file:
			idd=line.rstrip()
			idd=re.sub(".tsv$", "", idd)
			ids.append(idd)
	return ids

def proc(split_folder, infolder, out_top_folder):

	for split in range(10):
		train_ids=get_ids_from_filename("%s/%s/train.ids" % (split_folder, split))
		dev_ids=get_ids_from_filename("%s/%s/dev.ids" % (split_folder, split))
		test_ids=get_ids_from_filename("%s/%s/test.ids" % (split_folder, split))

		outfolder="%s/%s" % (out_top_folder, split)
		try:
			os.makedirs(outfolder)
		except:
			pass

		outTrainFile="%s/%s" % (outfolder, "train.conll")
		create_data(train_ids, infolder, outTrainFile)

		outTestFile="%s/%s" % (outfolder, "test.conll")
		create_data(test_ids, infolder, outTestFile)

		outDevFile="%s/%s" % (outfolder, "dev.conll")
		create_data(dev_ids, infolder, outDevFile)


# python scripts/create_crossval.py data/litbank_tenfold_splits data/original/conll/  data/litbank_tenfold_splits
filename=sys.argv[1]
infolder=sys.argv[2]
out_top_folder=sys.argv[3]
proc(filename, infolder, out_top_folder)