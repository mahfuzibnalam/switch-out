from transformers import MarianTokenizer, MarianMTModel
from typing import List
import sys

inpfile = sys.argv[1]
trgfile = sys.argv[2]
src_lang = 'en'

with open(inpfile) as inp:
	lines = inp.readlines()
lines = [l.strip() for l in lines[1:]]


langs = "es".split(',')

for trg in langs:
	print(f"Translating to {trg}...")
	lines2 = [f">>{trg}<< {l}" for l in lines]
	mname = f'Helsinki-NLP/opus-mt-en-ROMANCE'
	model = MarianMTModel.from_pretrained(mname)
	tok = MarianTokenizer.from_pretrained(mname)
	N = int(len(lines2)/10)
	with open(trgfile, 'w') as op:
		for i in range(N+1):
			if i%10==0:
				print(i)
			batch = tok.prepare_translation_batch(src_texts=lines2[10*i:10*(i+1)])
			gen = model.generate(**batch)
			words: List[str] = tok.batch_decode(gen, skip_special_tokens=True)
			op.write('\n'.join(words) + '\n')





