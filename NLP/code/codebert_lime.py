import torch
import pandas  as pd

from lime.lime_text import LimeTextExplainer

from transformers import RobertaTokenizer
from codebert_model import CodeBert
import os
import pickle


def predict_example(test_comment_list):

    USE_CUDA = torch.cuda.is_available()

    net = CodeBert()
    net.load_state_dict(torch.load(model_path))
    if(USE_CUDA):
        net.cuda()
    # tokenize
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    result_comments_id = tokenizer(test_comment_list,
                                    padding=True,
                                    truncation=True,
                                    max_length=120,
                                    return_tensors='pt')
    tokenizer_id = result_comments_id['input_ids']
    inputs = tokenizer_id
    batch_size = inputs.size(0)
    # initialize hidden state

    if(USE_CUDA):
        inputs = inputs.cuda()

    net.eval()
    with torch.no_grad():
        # get the output from the model
        output= net(inputs)
        output=torch.nn.Softmax(dim=1)(output)
        return output.cpu().detach().numpy()
        


batch_size = 32
model_path = "models/bert_model"
bert_model_class = "microsoft/codebert-base"
test_dataset_path = 'data/python_data/split_data/test.csv'

label_names = ['negative', 'positive']
explainer = LimeTextExplainer(class_names=label_names)

exps=[]

df = pd.read_csv(test_dataset_path)
example0 = df.iloc[0]['code']
example1 = df.iloc[1]['code']

with open("D:/2022summer/notebooks/modeexample1.txt","r") as f:
    for line in f:
        example2=line

# for example in [example0, example1]:
for example in [example2]:
    predictor = predict_example
    # predict_example(example)
    exp = explainer.explain_instance(example,predictor,num_samples=50)
    exps.append(exp)

pickle.dump(exps, open(os.path.join("lime_explainers","codebert_exp2.bin"), "wb"))




