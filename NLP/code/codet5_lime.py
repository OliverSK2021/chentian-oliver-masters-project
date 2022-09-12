from transformers import T5ForConditionalGeneration, RobertaTokenizer
import torch
import pandas as pd
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
import numpy as np
import pickle

model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")
# model_path = "D:/CodeT5/codebert/code/outputs/model_files/pytorch_model.bin"
model_path = "code/pytorch_model51.bin"
model.load_state_dict(torch.load(model_path))

tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")

test = pd.read_csv("D:/2022summer/data/clean_data/test.csv")
example0 = test.iloc[22]['new_code']
example1 = test.iloc[23]['new_code']
def prediction(example_list):
    logits=[]
    for example in example_list:
        input_ids = tokenizer(example, truncation=True, return_tensors="pt").input_ids
        generated_ids = model.generate(input_ids, max_length=512, output_scores=True, return_dict_in_generate=True)
        logit = generated_ids.scores[1][0][20:22]
        logit = F.softmax(logit, -1)
        logits.append(logit.tolist())
    return np.array(logits)

label_names = ['negative', 'positive']
explainer = LimeTextExplainer(class_names=label_names)
predictor = prediction
exps=[]
# for example in [example0,example1]:
examples=[]
for i in range(test.shape[0]):
    example = test.iloc[i]['new_code']
    # label = test.iloc[i]['label']
    # print(predictor([example]))
    # exp = explainer.explain_instance(example,predictor,num_samples=50)
    # exps.append(exp)
    examples.append(example)
print(predictor(examples))

pickle.dump(exps, open("code/exps.bin", "wb"))


