import torch
import torch.nn as nn
from transformers import RobertaModel



class Bert_lstm(nn.Module):
    def __init__(self, hidden_dim, output_size,n_layers,bidirectional=True, drop_prob=0.5):
        super(Bert_lstm, self).__init__()
 
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        #Bert ----------------重点，bert模型需要嵌入到自定义模型里面
        self.bert=RobertaModel.from_pretrained("microsoft/codebert-small")
        for param in self.bert.parameters():
            param.requires_grad = True
        
        # dropout layer
        self.dropout = nn.Dropout(drop_prob)
           
                
        # LSTM layers
        self.lstm = nn.LSTM(input_size = 768, 
            hidden_size = self.hidden_dim, 
            num_layers = self.n_layers,
            batch_first=True,
            bidirectional=bool(bidirectional))

        # self.lstm = nn.LSTM(768,self.hidden_dim,bidirectional=True)


        # linear and sigmoid layers
        if bidirectional:
            # self.fc = nn.Linear(hidden_dim*2, output_size)
            self.fc = nn.Linear(hidden_dim*2, output_size)
        else:
            self.fc = nn.Linear(hidden_dim, output_size)
          
        #self.sig = nn.Sigmoid()
 
    def forward(self, x, hidden):
        batch_size = x.size(0)
        #生成bert字向量
        x=self.bert(x)[0]     #bert 字向量
        
        # lstm_out
        x = x.float()
        lstm_out, (hidden_last,cn_last) = self.lstm(x, hidden)
        
        # 修改 双向的需要单独处理
        if self.bidirectional:
            # 正向最后一层，最后一个时刻
            hidden_last_L=hidden_last[-2]
            # 反向最后一层，最后一个时刻
            hidden_last_R=hidden_last[-1]
            # 进行拼接
            hidden_last_out=torch.cat([hidden_last_L,hidden_last_R],dim=-1)
        else:
            hidden_last_out=hidden_last[-1]   #[32, 384]
            
            
        # dropout and fully-connected layer
        out = self.dropout(hidden_last_out)
        out = self.fc(out)
        
        return out
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        
        number = 1
        if self.bidirectional:
            number = 2
        USE_CUDA = torch.cuda.is_available()
        if (USE_CUDA):
            hidden = (weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_().float().cuda(),
                      weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_().float().cuda()
                     )
        else:
            hidden = (weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_().float(),
                      weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_().float()
                     )
        
        return hidden

