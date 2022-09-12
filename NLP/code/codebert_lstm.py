import argparse
import torch
import torch.nn as nn
from transformers import RobertaTokenizer


import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from lstm_model import Bert_lstm
import os

np.random.seed(2022)
torch.manual_seed(2022)
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    torch.cuda.manual_seed(2022)




path = "D:/2022summer/data/python_data/split_data/"
train = pd.read_csv(os.path.join(path,'train.csv'))
test = pd.read_csv(os.path.join(path,'test.csv'))
valid = pd.read_csv(os.path.join(path,'valid.csv'))


tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

train_result_comments = list(train['code'].values)
X_train = tokenizer(train_result_comments,padding=True,truncation=True,max_length=2000,return_tensors='pt')['input_ids']
y_train = torch.from_numpy(train['label'].values).float()

test_result_comments = list(test['code'].values)
X_test = tokenizer(test_result_comments,padding=True,truncation=True,max_length=2000,return_tensors='pt')['input_ids']
y_test = torch.from_numpy(test['label'].values).float()

valid_result_comments = list(valid['code'].values)
X_valid = tokenizer(valid_result_comments,padding=True,truncation=True,max_length=2000,return_tensors='pt')['input_ids']
y_valid = torch.from_numpy(valid['label'].values).float()


# create Tensor datasets
train_data = TensorDataset(X_train, y_train)
valid_data = TensorDataset(X_valid, y_valid)
test_data = TensorDataset(X_test,y_test)

# dataloaders
batch_size = 32

# make sure the SHUFFLE your training data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size,drop_last=True)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size,drop_last=True)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size,drop_last=True)


def train_model(hidden_dim, output_size, n_layers, bidirectional, save_path):

    net = Bert_lstm(
                hidden_dim, 
                output_size,
                n_layers, 
                bidirectional)

    #print(net)

    # loss and optimization functions
    lr=2e-5
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # training params
    epochs = 10
    print_every = 7
    clip=5 # gradient clipping
    
    # move model to GPU, if available
    if(USE_CUDA):
        net.cuda()

    net.train()
    # train for some number of epochs
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)
        counter = 0
    
        # batch loop
        for inputs, labels in train_loader:
            counter += 1
            
            if(USE_CUDA):
                inputs, labels = inputs.cuda(), labels.cuda()
            h = tuple([each.data for each in h])
            net.zero_grad()
            output= net(inputs, h)
            loss = criterion(output.squeeze(), labels.long())
            loss.backward()
            optimizer.step()
    
            # loss stats
            if counter % print_every == 0:
                net.eval()
                with torch.no_grad():
                    val_h = net.init_hidden(batch_size)
                    val_losses = []
                    for inputs, labels in valid_loader:
                        val_h = tuple([each.data for each in val_h])

                        if(USE_CUDA):
                            inputs, labels = inputs.cuda(), labels.cuda()

                        output = net(inputs, val_h)
                        val_loss = criterion(output.squeeze(), labels.long())

                        val_losses.append(val_loss.item())
    
                net.train()
                print("Epoch: {}/{}...".format(e+1, epochs),
                    "Step: {}...".format(counter),
                    "Loss: {:.6f}...".format(loss.item()),
                    "Val Loss: {:.6f}".format(np.mean(val_losses)))
    torch.save(net.state_dict(), save_path)


def test_model(hidden_dim, output_size, n_layers, bidirectional, save_path):

    net = Bert_lstm(
                hidden_dim, 
                output_size,
                n_layers, 
                bidirectional)
    net.load_state_dict(torch.load(save_path))
    if(USE_CUDA):
        net.cuda()
    criterion = nn.CrossEntropyLoss()
    test_losses = [] # track loss
    # num_correct = 0
    
    # init hidden state
    h = net.init_hidden(batch_size)
    
    net.eval()
    # iterate over test data
    preds=[]
    labels_list=[]
    for inputs, labels in test_loader:
        h = tuple([each.data for each in h])
        if(USE_CUDA):
            inputs, labels = inputs.cuda(), labels.cuda()
        output = net(inputs, h)
        test_loss = criterion(output.squeeze(), labels.long())
        test_losses.append(test_loss.item())
        
        output=torch.nn.Softmax(dim=1)(output)
        pred=torch.max(output, 1)[1]
        
        # compare predictions to true label
        # correct_tensor = pred.eq(labels.long().view_as(pred))
        # correct = np.squeeze(correct_tensor.numpy()) if not USE_CUDA else np.squeeze(correct_tensor.cpu().numpy())
        # num_correct += np.sum(correct)
        preds+=pred.tolist()
        labels_list+=labels.int().tolist()
    if len(preds) < len(test_loader.dataset):
        for i in range(len(preds),len(test_loader.dataset)):
            example = tokenizer.decode(X_test[i])
            label = y_test[i]
            labels_list.append(label.int().tolist())
            pred = predict_example(example, hidden_dim, output_size, n_layers, bidirectional, save_path)
            preds.append(pred)
    print("Test loss: {:.3f}".format(np.mean(test_losses)))
    # accuracy over all test data
    # test_acc = num_correct/len(test_loader.dataset)
    # print("Test accuracy: {:.3f}".format(test_acc))
    prediction = [preds,labels_list]
    torch.save(prediction, save_path+"_prediction.bin")


def predict_example(test_comment_list, hidden_dim, output_size, n_layers, bidirectional, save_path):

    net = Bert_lstm( 
                hidden_dim, 
                output_size,
                n_layers, 
                bidirectional)
    net.load_state_dict(torch.load(save_path))
    if(USE_CUDA):
        net.cuda()
    result_comments=test_comment_list   #预处理去掉标点符号
    #转换为字id
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-small")
    result_comments_id = tokenizer(result_comments,
                                    padding=True,
                                    truncation=True,
                                    max_length=120,
                                    return_tensors='pt')
    tokenizer_id = result_comments_id['input_ids']
    # print(tokenizer_id.shape)
    inputs = tokenizer_id
    batch_size = inputs.size(0)
    # batch_size = 32
    # initialize hidden state
    h = net.init_hidden(batch_size)

    if(USE_CUDA):
        inputs = inputs.cuda()

    net.eval()
    with torch.no_grad():
        # get the output from the model
        output= net(inputs, h)
        output=torch.nn.Softmax(dim=1)(output)
        pred=torch.max(output, 1)[1]
        # printing output value, before rounding
        print('预测概率为: {:.6f}'.format(torch.max(output, 1)[0].item()))
        if(pred.item()==1):
            print("预测结果为:正向")
        else:
            print("预测结果为:负向")
    return pred.item()


def main():
    output_size = 2
    hidden_dim = 384   #768/2
    n_layers = 2
    bidirectional = True
    save_path = "D:/2022summer/models/codebert_lstm"
    batch_size = 32
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--test", action='store_true',
                        help="Whether to run testing.")
    parser.add_argument("--predict_example", action='store_true',
                        help="Whether to prediction an example.")
    args = parser.parse_args()

    if args.train:
        train_model(hidden_dim, output_size, n_layers, bidirectional, save_path)
    if args.test:
        test_model(hidden_dim, output_size, n_layers, bidirectional, save_path)
    if args.predict_example:
        example = tokenizer.decode(X_test[0])
        # label = y_test[0]
        predict_example(example, hidden_dim, output_size, n_layers, bidirectional, save_path)


if __name__ == '__main__':
    main()