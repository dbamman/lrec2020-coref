import sys

def gen(path_to_scorer, train_out_file, pred_out_file):
	train_out=open(train_out_file, "w", encoding="utf-8")
	pred_out=open(pred_out_file, "w", encoding="utf-8")

	for i in range(10):
		train_out.write ("python3 scripts/bert_coref.py -m train -w models/crossval/%s.model -t data/litbank_tenfold_splits/%s/train.conll -v data/litbank_tenfold_splits/%s/dev.conll -o preds/crossval/%s.dev.pred -s %s> logs/crossval/%s.log 2>&1\n" % (i, i, i, i, path_to_scorer, i))


		pred_out.write("python3 scripts/bert_coref.py -m predict -w models/crossval/%s.model -v data/litbank_tenfold_splits/%s/test.conll -o preds/crossval/%s.goldmentions.test.preds -s %s\n" % (i, i, i, path_to_scorer))

	train_out.close()
	pred_out.close()

gen(sys.argv[1], sys.argv[2], sys.argv[3])