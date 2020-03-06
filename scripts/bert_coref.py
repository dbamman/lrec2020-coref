import re
import os
from collections import Counter
import sys
import argparse

import pytorch_pretrained_bert
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertConfig
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

import torch
from torch import nn
import torch.optim as optim
import numpy as np
import random
import calc_coref_metrics

from torch.optim.lr_scheduler import ExponentialLR

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

bert_dim=768
HIDDEN_DIM=200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMTagger(BertPreTrainedModel):

	def __init__(self, config, freeze_bert=False):
		super(LSTMTagger, self).__init__(config)

		hidden_dim=HIDDEN_DIM
		self.hidden_dim=hidden_dim

		self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False, do_basic_tokenize=False)
		self.bert = BertModel.from_pretrained("bert-base-cased")
		self.bert.eval()

		if freeze_bert:
			for param in self.bert.parameters():
				param.requires_grad = False

		self.distance_embeddings = nn.Embedding(11, 20)
		self.sent_distance_embeddings = nn.Embedding(11, 20)
		self.nested_embeddings = nn.Embedding(2, 20)
		self.gender_embeddings = nn.Embedding(3, 20)
		self.width_embeddings = nn.Embedding(12, 20)
		self.quote_embeddings = nn.Embedding(3, 20)

		self.lstm = nn.LSTM(4*bert_dim, hidden_dim, bidirectional=True, batch_first=True)

		self.attention1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
		self.attention2 = nn.Linear(hidden_dim * 2, 1)
		self.mention_mention1 = nn.Linear( (3 * 2 * hidden_dim + 20 + 20) * 3 + 20 + 20 + 20 + 20, 150)
		self.mention_mention2 = nn.Linear(150, 150)
		self.mention_mention3 = nn.Linear(150, 1)

		self.unary1 = nn.Linear(3 * 2 * hidden_dim + 20 + 20, 150)
		self.unary2 = nn.Linear(150, 150)
		self.unary3 = nn.Linear(150, 1)

		self.drop_layer_020 = nn.Dropout(p=0.2)
		self.tanh = nn.Tanh()

		self.apply(self.init_bert_weights)


	def get_mention_reps(self, input_ids=None, attention_mask=None, starts=None, ends=None, index=None, widths=None, quotes=None, matrix=None, transforms=None, doTrain=True):

		starts=starts.to(device)
		ends=ends.to(device)
		widths=widths.to(device)
		quotes=quotes.to(device)

		input_ids = input_ids.to(device)
		attention_mask = attention_mask.to(device)
		transforms = transforms.to(device)

		# matrix specifies which token positions (cols) are associated with which mention spans (row)
		matrix=matrix.to(device) # num_sents x max_ents x max_words

		# index specifies the location of the mentions in each sentence (which vary due to padding)
		index=index.to(device)

		sequence_outputs, pooled_outputs = self.bert(input_ids, token_type_ids=None, attention_mask=attention_mask)
		all_layers = torch.cat((sequence_outputs[-1], sequence_outputs[-2], sequence_outputs[-3], sequence_outputs[-4]), 2)
		embeds=torch.matmul(transforms,all_layers)
		lstm_output, _ = self.lstm(embeds) # num_sents x max_words x 2 * hidden_dim

		###########
		# ATTENTION OVER MENTION
		###########

		attention_weights=self.attention2(self.tanh(self.attention1(lstm_output))) # num_sents x max_words x 1
		attention_weights=torch.exp(attention_weights)
		
		attx=attention_weights.squeeze(-1).unsqueeze(1).expand_as(matrix)
		summer=attx*matrix

		val=matrix*summer # num_sents x max_ents x max_words
		
		val=val/torch.sum(1e-8+val,dim=2).unsqueeze(-1)

		attended=torch.matmul(val, lstm_output) # num_sents x max_ents x 2 * hidden_dim

		attended=attended.view(-1,2*self.hidden_dim)

		lstm_output=lstm_output.contiguous()
		position_output=lstm_output.view(-1, 2*self.hidden_dim)
		
		# starts = token position of beginning of mention in flattened token list
		start_output=torch.index_select(position_output, 0, starts)
		# ends = token position of end of mention in flattened token list
		end_output=torch.index_select(position_output, 0, ends)

		# index = index of entity in flattened list of attended mention representations
		mentions=torch.index_select(attended, 0, index)

		width_embeds=self.width_embeddings(widths)
		quote_embeds=self.quote_embeddings(quotes)

		span_representation=torch.cat((start_output, end_output, mentions, width_embeds, quote_embeds), 1)

		if doTrain:
			return span_representation
		else:
			# detach tensor from computation graph at test time or memory will blow up
			return span_representation.detach()


	def forward(self, matrix, index, truth=None, names=None, token_positions=None, starts=None, ends=None, widths=None, input_ids=None, attention_mask=None, transforms=None, quotes=None):

		doTrain=False
		if truth is not None:
			doTrain=True

		zeroTensor=torch.FloatTensor([0]).to(device)

		all_starts=None
		all_ends=None

		span_representation=None

		all_all=[]
		for b in range(len(matrix)):

			span_reps=self.get_mention_reps(input_ids=input_ids[b], attention_mask=attention_mask[b], starts=starts[b], ends=ends[b], index=index[b], widths=widths[b], quotes=quotes[b], transforms=transforms[b], matrix=matrix[b], doTrain=doTrain)
			if b == 0:
				span_representation=span_reps
				all_starts=starts[b]
				all_ends=ends[b]

			else:

				span_representation=torch.cat((span_representation, span_reps), 0)
	
				all_starts=torch.cat((all_starts, starts[b]), 0)
				all_ends=torch.cat((all_ends, ends[b]), 0)

		all_starts=all_starts.to(device)
		all_ends=all_ends.to(device)
		
		num_mentions,=all_starts.shape

		running_loss=0

		curid=-1

		curid+=1

		assignments=[]

		seen={}

		ch=0

		token_positions=np.array(token_positions)

		mention_index=np.arange(num_mentions)

		unary_scores=self.unary3(self.tanh(self.drop_layer_020(self.unary2(self.tanh(self.drop_layer_020(self.unary1(span_representation)))))))

		for i in range(num_mentions):

			if i == 0:
				# the first mention must start a new entity; this doesn't affect training (since the loss must be 0) so we can skip it.
				if truth is None:

						assignment=curid
						curid+=1
						assignments.append(assignment)
				
				continue

			MAX_PREVIOUS_MENTIONS=300

			first=0
			if truth is None:
				if len(names[i]) == 1 and names[i][0].lower() in {"he", "his", "her", "she", "him", "they", "their", "them", "it", "himself", "its", "herself", "themselves"}:
					MAX_PREVIOUS_MENTIONS=20

				first=i-MAX_PREVIOUS_MENTIONS
				if first < 0:
					first=0

			targets=span_representation[first:i]
			cp=span_representation[i].expand_as(targets)
			
			dists=[]
			nesteds=[]

			# get distance in mentions
			distances=i-mention_index[first:i]
			dists=vec_get_distance_bucket(distances)
			dists=torch.LongTensor(dists).to(device)
			distance_embeds=self.distance_embeddings(dists)

			# get distance in sentences
			sent_distances=token_positions[i]-token_positions[first:i]
			sent_dists=vec_get_distance_bucket(sent_distances)
			sent_dists=torch.LongTensor(sent_dists).to(device)
			sent_distance_embeds=self.sent_distance_embeddings(sent_dists)

			# is the current mention nested within a previous one?
			res1=(all_starts[first:i] <= all_starts[i]) 
			res2=(all_ends[i] <= all_ends[first:i])

			nesteds=(res1*res2).long()
			nesteds_embeds=self.nested_embeddings(nesteds)

			res1=(all_starts[i] <= all_starts[first:i]) 
			res2=(all_ends[first:i] <= all_ends[i])

			nesteds=(res1*res2).long()
			nesteds_embeds2=self.nested_embeddings(nesteds)

			elementwise=cp*targets
			concat=torch.cat((cp, targets, elementwise, distance_embeds, sent_distance_embeds, nesteds_embeds, nesteds_embeds2), 1)

			preds=self.mention_mention3(self.tanh(self.drop_layer_020(self.mention_mention2(self.tanh(self.drop_layer_020(self.mention_mention1(concat)))))))

			preds=preds + unary_scores[i] + unary_scores[first:i]

			preds=preds.squeeze(-1)

			if truth is not None:
	
				# zero is the score for the dummy antecedent/new entity
				preds=torch.cat((preds, zeroTensor))
	
				golds_sum=0.
				preds_sum=torch.logsumexp(preds, 0)

				if len(truth[i]) == 1 and truth[i][-1] not in seen:
					golds_sum=0.
					seen[truth[i][-1]]=1
				else:
					golds=torch.index_select(preds, 0, torch.LongTensor(truth[i]).to(device))
					golds_sum=torch.logsumexp(golds, 0)
				
				# want to maximize (golds_sum-preds_sum), so minimize (preds_sum-golds_sum)
				diff=preds_sum-golds_sum

				running_loss+=diff

			else:

				assignment=None

				if i == 0:
					assignment=curid
					curid+=1

				else:

					arg_sorts=torch.argsort(preds, descending=True)
					k=0
					while k < len(arg_sorts):
						cand_idx=arg_sorts[k]
						if preds[cand_idx] > 0:
							cand_assignment=assignments[cand_idx+first]
							assignment=cand_assignment
							ch+=1
							break

						else:
							assignment=curid
							curid+=1
							break

						k+=1


				assignments.append(assignment)

		if truth is not None:
			return running_loss
		else:
			return assignments


def get_mention_width_bucket(dist):
	if dist < 10:
		return dist + 1

	return 11

def get_distance_bucket(dist):
	if dist < 5:
		return dist+1

	elif dist >= 5 and dist <= 7:
		return 6
	elif dist >= 8 and dist <= 15:
		return 7
	elif dist >= 16 and dist <= 31:
		return 8
	elif dist >= 32 and dist <= 63:
		return 9

	return 10

vec_get_distance_bucket=np.vectorize(get_distance_bucket)

def get_inquote(start, end, sent):

	inQuote=False
	quotes=[]

	for token in sent:
		if token == "“" or token == "\"":
			if inQuote == True:
				inQuote=False
			else:
				inQuote=True

		quotes.append(inQuote)

	for i in range(start, end+1):
		if quotes[i] == True:
			return 1

	return 0


def print_conll(name, sents, all_ents, assignments, out):

	doc_id, part_id=name

	mapper=[]
	idd=0
	for ent in all_ents:
		mapper_e=[]
		for e in ent:
			mapper_e.append(idd)
			idd+=1
		mapper.append(mapper_e)

	out.write("#begin document (%s); part %s\n" % (doc_id, part_id))
	
	for s_idx, sent in enumerate(sents):
		ents=all_ents[s_idx]
		for w_idx, word in enumerate(sent):
			if w_idx == 0 or w_idx == len(sent)-1:
				continue

			label=[]
			for idx, (start, end) in enumerate(ents):
				if start == w_idx and end == w_idx:
					eid=assignments[mapper[s_idx][idx]]
					label.append("(%s)" % eid)
				elif start == w_idx:
					eid=assignments[mapper[s_idx][idx]]
					label.append("(%s" % eid)
				elif end == w_idx:
					eid=assignments[mapper[s_idx][idx]]
					label.append("%s)" % eid)

			out.write("%s\t%s\t%s\t%s\t_\t_\t_\t_\t_\t_\t_\t_\t%s\n" % (doc_id, part_id, w_idx-1, word, '|'.join(label)))

		if len(sent) > 2:
			out.write("\n")

	out.write("#end document\n")


def test(model, test_all_docs, test_all_ents, test_all_named_ents, test_all_max_words, test_all_max_ents, test_doc_names, outfile, iterr, goldFile, path_to_scorer, doTest=False):

	out=open(outfile, "w", encoding="utf-8")

	# for each document
	for idx in range(len(test_all_docs)):

		d,p=test_doc_names[idx]
		d=re.sub("/", "_", d)
		test_doc=test_all_docs[idx]
		test_ents=test_all_ents[idx]

		max_words=test_all_max_words[idx]
		max_ents=test_all_max_ents[idx]

		names=[]
		for n_idx, sent in enumerate(test_ents):
			for ent in sent:
				name=test_doc[n_idx][ent[0]:ent[1]+1]
				names.append(name)

		named_index={}
		for sidx, sentence in enumerate(test_all_named_ents[idx]):
			for start, end, _ in sentence:
				named_index[(sidx, start, end)]=1

		is_named=[]

		for sidx, sentence in enumerate(test_all_ents[idx]):
			for (start, end) in sentence:
				if (sidx, start, end) in named_index:
					is_named.append(1)
				else:
					is_named.append(0)

		test_matrix, test_index, test_token_positions, test_ent_spans, test_starts, test_ends, test_widths, test_data, test_masks, test_transforms, test_quotes=get_data(model, test_doc, test_ents, max_ents, max_words)

		assignments=model.forward(test_matrix, test_index, names=names, token_positions=test_token_positions, starts=test_starts, ends=test_ends, widths=test_widths, input_ids=test_data, attention_mask=test_masks, transforms=test_transforms, quotes=test_quotes)
		print_conll(test_doc_names[idx], test_doc, test_ents, assignments, out)

	out.close()

	if doTest:
		print("Goldfile: %s" % goldFile)
		print("Predfile: %s" % outfile)
		
		bcub_f, avg=calc_coref_metrics.get_conll(path_to_scorer, gold=goldFile, preds=outfile)
		print("Iter %s, Average F1: %.3f, bcub F1: %s" % (iterr, avg, bcub_f))
		sys.stdout.flush()
		return avg

def get_matrix(list_of_entities, max_words, max_ents):
	matrix=np.zeros((max_ents, max_words))
	for idx, (start, end) in enumerate(list_of_entities):
		for i in range(start, end+1):
			matrix[idx,i]=1
	return matrix


def get_data(model, doc, ents, max_ents, max_words):

	batchsize=128

	token_positions=[]
	ent_spans=[]
	persons=[]
	inquotes=[]

	batch_matrix=[]
	matrix=[]

	max_words_batch=[]
	max_ents_batch=[]

	max_w=1
	max_e=1

	sent_count=0
	for idx, sent in enumerate(doc):
		
		if len(sent) > max_w:
			max_w=len(sent)
		if len(ents[idx]) > max_e:
			max_e = len(ents[idx])

		sent_count+=1

		if sent_count == batchsize:
			max_words_batch.append(max_w)
			max_ents_batch.append(max_e)
			sent_count=0
			max_w=0
			max_e=0

	if sent_count > 0:
		max_words_batch.append(max_w)
		max_ents_batch.append(max_e)

	batch_count=0
	for idx, sent in enumerate(doc):
		matrix.append(get_matrix(ents[idx], max_words_batch[batch_count], max_ents_batch[batch_count]))

		if len(matrix) == batchsize:
			batch_matrix.append(torch.FloatTensor(matrix))
			matrix=[]
			batch_count+=1

	if len(matrix) > 0:
		batch_matrix.append(torch.FloatTensor(matrix))


	batch_index=[]
	batch_quotes=[]

	batch_ent_spans=[]

	index=[]
	abs_pos=0
	sent_count=0

	b=0
	for idx, sent in enumerate(ents):

		for i in range(len(sent)):
			index.append(sent_count*max_ents_batch[b] + i)
			s,e=sent[i]
			token_positions.append(idx)
			ent_spans.append(e-s)
			phrase=' '.join(doc[idx][s:e+1])

			inquotes.append(get_inquote(s, e, doc[idx]))


		abs_pos+=len(doc[idx])

		sent_count+=1

		if sent_count == batchsize:
			batch_index.append(torch.LongTensor(index))
			batch_quotes.append(torch.LongTensor(inquotes))
			batch_ent_spans.append(ent_spans)

			index=[]
			inquotes=[]
			ent_spans=[]
			sent_count=0
			b+=1

	if sent_count > 0:
		batch_index.append(torch.LongTensor(index))
		batch_quotes.append(torch.LongTensor(inquotes))
		batch_ent_spans.append(ent_spans)

	all_masks=[]
	all_transforms=[]
	all_data=[]

	batch_masks=[]
	batch_transforms=[]
	batch_data=[]

	# get ids and pad sentence
	for sent in doc:
		tok_ids=[]
		input_mask=[]
		transform=[]

		all_toks=[]
		n=0
		for idx, word in enumerate(sent):
			toks=model.tokenizer.tokenize(word)
			all_toks.append(toks)
			n+=len(toks)


		cur=0
		for idx, word in enumerate(sent):

			toks=all_toks[idx]
			ind=list(np.zeros(n))
			for j in range(cur,cur+len(toks)):
				ind[j]=1./len(toks)
			cur+=len(toks)
			transform.append(ind)

			tok_id=model.tokenizer.convert_tokens_to_ids(toks)
			assert len(tok_id) == len(toks)
			tok_ids.extend(tok_id)

			input_mask.extend(np.ones(len(toks)))

			token=word.lower()

		all_masks.append(input_mask)
		all_data.append(tok_ids)
		all_transforms.append(transform)

		if len(all_masks) == batchsize:
			batch_masks.append(all_masks)
			batch_data.append(all_data)
			batch_transforms.append(all_transforms)

			all_masks=[]
			all_data=[]
			all_transforms=[]

	if len(all_masks) > 0:
		batch_masks.append(all_masks)
		batch_data.append(all_data)
		batch_transforms.append(all_transforms)


	for b in range(len(batch_data)):

		max_len = max([len(sent) for sent in batch_data[b]])

		for j in range(len(batch_data[b])):
			
			blen=len(batch_data[b][j])

			for k in range(blen, max_len):
				batch_data[b][j].append(0)
				batch_masks[b][j].append(0)
				for z in range(len(batch_transforms[b][j])):
					batch_transforms[b][j][z].append(0)

			for k in range(len(batch_transforms[b][j]), max_words_batch[b]):
				batch_transforms[b][j].append(np.zeros(max_len))

		batch_data[b]=torch.LongTensor(batch_data[b])
		batch_transforms[b]=torch.FloatTensor(batch_transforms[b])
		batch_masks[b]=torch.FloatTensor(batch_masks[b])
		
	tok_pos=0
	starts=[]
	ends=[]
	widths=[]

	batch_starts=[]
	batch_ends=[]
	batch_widths=[]

	sent_count=0
	b=0
	for idx, sent in enumerate(ents):

		for i in range(len(sent)):

			s,e=sent[i]

			starts.append(tok_pos+s)
			ends.append(tok_pos+e)
			widths.append(get_mention_width_bucket(e-s))

		sent_count+=1
		tok_pos+=max_words_batch[b]

		if sent_count == batchsize:
			batch_starts.append(torch.LongTensor(starts))
			batch_ends.append(torch.LongTensor(ends))
			batch_widths.append(torch.LongTensor(widths))

			starts=[]
			ends=[]
			widths=[]
			tok_pos=0
			sent_count=0
			b+=1

	if sent_count > 0:
		batch_starts.append(torch.LongTensor(starts))
		batch_ends.append(torch.LongTensor(ends))
		batch_widths.append(torch.LongTensor(widths))


	return batch_matrix, batch_index, token_positions, ent_spans, batch_starts, batch_ends, batch_widths, batch_data, batch_masks, batch_transforms, batch_quotes


def get_ant_labels(all_doc_sents, all_doc_ents):

	max_words=0
	max_ents=0
	mention_id=0

	big_ents={}

	doc_antecedent_labels=[]
	big_doc_ents=[]

	for idx, sent in enumerate(all_doc_sents):
		if len(sent) > max_words:
			max_words=len(sent)

		this_sent_ents=[]
		all_sent_ents=sorted(all_doc_ents[idx], key=lambda x: (x[0], x[1]))
		if len(all_sent_ents) > max_ents:
			max_ents=len(all_sent_ents)

		for (w_idx_start, w_idx_end, eid) in all_sent_ents:

			this_sent_ents.append((w_idx_start, w_idx_end))

			coref={}
			if eid in big_ents:
				coref=big_ents[eid]
			else:
				coref={mention_id:1}

			vals=sorted(coref.keys())

			if eid not in big_ents:
				big_ents[eid]={}

			big_ents[eid][mention_id]=1
			mention_id+=1

			doc_antecedent_labels.append(vals)

		big_doc_ents.append(this_sent_ents)


	return doc_antecedent_labels, big_doc_ents, max_words, max_ents


def read_conll(filename, model=None):

	docid=None
	partID=None

	# collection
	all_sents=[]
	all_ents=[]
	all_antecedent_labels=[]
	all_max_words=[]
	all_max_ents=[]
	all_doc_names=[]

	all_named_ents=[]


	# for one doc
	all_doc_sents=[]
	all_doc_ents=[]
	all_doc_named_ents=[]

	# for one sentence
	sent=[]
	ents=[]
	sent.append("[SEP]")
	sid=0

	named_ents=[]
	cur_tokens=0
	max_allowable_tokens=400
	cur_tid=0
	open_count=0
	with open(filename, encoding="utf-8") as file:
		for line in file:
			if line.startswith("#begin document"):

				all_doc_ents=[]
				all_doc_sents=[]

				all_doc_named_ents=[]

				open_ents={}
				open_named_ents={}

				sid=0
				docid=None
				matcher=re.match("#begin document \((.*)\); part (.*)$", line.rstrip())
				if matcher != None:
					docid=matcher.group(1)
					partID=matcher.group(2)

				print(docid)

			elif line.startswith("#end document"):

				all_sents.append(all_doc_sents)
				
				doc_antecedent_labels, big_ents, max_words, max_ents=get_ant_labels(all_doc_sents, all_doc_ents)

				all_ents.append(big_ents)

				all_named_ents.append(all_doc_named_ents)

				all_antecedent_labels.append(doc_antecedent_labels)
				all_max_words.append(max_words+1)
				all_max_ents.append(max_ents+1)
				
				all_doc_names.append((docid,partID))

			else:

				parts=re.split("\s+", line.rstrip())
				if len(parts) < 2 or (cur_tokens >= max_allowable_tokens and open_count == 0):
		
					sent.append("[CLS]")
					all_doc_sents.append(sent)
					ents=sorted(ents, key=lambda x: (x[0], x[1]))

					all_doc_ents.append(ents)

					all_doc_named_ents.append(named_ents)

					ents=[]
					named_ents=[]
					sent=[]
					sent.append("[SEP]")
					sid+=1

					cur_tokens=0

					cur_tid=0

					if len(parts) < 2:
						continue

				# +1 to account for initial [SEP]
				tid=cur_tid+1
				token=parts[3]
				coref=parts[-1].split("|")
				b_toks=model.tokenizer.tokenize(token)
				cur_tokens+=len(b_toks)
				cur_tid+=1

				for c in coref:
					if c.startswith("(") and c.endswith(")"):
						c=re.sub("\(", "", c)
						c=int(re.sub("\)", "", c))

						ents.append((tid, tid, c))

					elif c.startswith("("):
						c=int(re.sub("\(", "", c))

						if c not in open_ents:
							open_ents[c]=[]
						open_ents[c].append(tid)
						open_count+=1

					elif c.endswith(")"):
						c=int(re.sub("\)", "", c))

						assert c in open_ents

						start_tid=open_ents[c].pop()
						open_count-=1

						ents.append((start_tid, tid, c))

				ner=parts[10].split("|")

				for c in ner:
					try:
						if c.startswith("(") and c.endswith(")"):
							c=re.sub("\(", "", c)
							c=int(re.sub("\)", "", c))

							named_ents.append((tid, tid, c))

						elif c.startswith("("):
							c=int(re.sub("\(", "", c))

							if c not in open_named_ents:
								open_named_ents[c]=[]
							open_named_ents[c].append(tid)

						elif c.endswith(")"):
							c=int(re.sub("\)", "", c))

							assert c in open_named_ents

							start_tid=open_named_ents[c].pop()

							named_ents.append((start_tid, tid, c))
					except:
						pass

				sent.append(token)

	return all_sents, all_ents, all_named_ents, all_antecedent_labels, all_max_words, all_max_ents, all_doc_names


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-t','--trainData', help='Folder containing train data', required=False)
	parser.add_argument('-v','--valData', help='Folder containing test data', required=False)
	parser.add_argument('-m','--mode', help='mode {train, predict}', required=False)
	parser.add_argument('-w','--model', help='modelFile (to write to or read from)', required=False)
	parser.add_argument('-o','--outFile', help='outFile', required=False)
	parser.add_argument('-s','--path_to_scorer', help='Path to coreference scorer', required=False)

	args = vars(parser.parse_args())

	mode=args["mode"]
	modelFile=args["model"]
	valData=args["valData"]
	outfile=args["outFile"]
	path_to_scorer=args["path_to_scorer"]
	
	cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(0))

	model = LSTMTagger.from_pretrained('bert-base-cased',
			  cache_dir=cache_dir,
			  freeze_bert=True)

	config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
		num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

	model.to(device)

	optimizer = optim.Adam(model.parameters(), lr=0.001)
	lr_scheduler=ExponentialLR(optimizer, gamma=0.999)

	if mode == "train":

		trainData=args["trainData"]

		all_docs, all_ents, all_named_ents, all_truth, all_max_words, all_max_ents, doc_ids=read_conll(trainData, model)
		test_all_docs, test_all_ents, test_all_named_ents, test_all_truth, test_all_max_words, test_all_max_ents, test_doc_ids=read_conll(valData, model)


		best_f1=0.
		cur_steps=0

		best_idx=0
		patience=10

		for i in range(100):

			model.train()
			bigloss=0.
			for idx in range(len(all_docs)):

				if idx % 10 == 0:
					print(idx, "/", len(all_docs))
					sys.stdout.flush()
				max_words=all_max_words[idx]
				max_ents=all_max_ents[idx]

				matrix, index, token_positions, ent_spans, starts, ends, widths, input_ids, masks, transforms, quotes=get_data(model, all_docs[idx], all_ents[idx], max_ents, max_words)

				if max_ents > 1:
					model.zero_grad()
					loss=model.forward(matrix, index, truth=all_truth[idx], names=None, token_positions=token_positions, starts=starts, ends=ends, widths=widths, input_ids=input_ids, attention_mask=masks, transforms=transforms, quotes=quotes)
					loss.backward()
					optimizer.step()
					cur_steps+=1
					if cur_steps % 100 == 0:
						lr_scheduler.step()
				bigloss+=loss.item()

			print(bigloss)

			model.eval()
			doTest=False
			if i >= 2:
				doTest=True
			
			avg_f1=test(model, test_all_docs, test_all_ents, test_all_named_ents, test_all_max_words, test_all_max_ents, test_doc_ids, outfile, i, valData, path_to_scorer, doTest=doTest)

			if doTest:
				if avg_f1 > best_f1:
					torch.save(model.state_dict(), modelFile)
					print("Saving model ... %.3f is better than %.3f" % (avg_f1, best_f1))
					best_f1=avg_f1
					best_idx=i

				if i-best_idx > patience:
					print ("Stopping training at epoch %s" % i)
					break

	elif mode == "predict":

		model.load_state_dict(torch.load(modelFile, map_location=device))
		model.eval()
		test_all_docs, test_all_ents, test_all_named_ents, test_all_truth, test_all_max_words, test_all_max_ents, test_doc_ids=read_conll(valData, model=model)

		test(model, test_all_docs, test_all_ents, test_all_named_ents, test_all_max_words, test_all_max_ents, test_doc_ids, outfile, 0, valData, path_to_scorer, doTest=True)

