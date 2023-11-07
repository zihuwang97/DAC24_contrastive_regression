import torch
import torch.nn as nn

class encoder(nn.Module):
    def __init__(self, input_feature_num, hdn_size):
        super(encoder, self).__init__()
        """
        2-layer encoder; 
        Activation func=LeakyReLU;
        --------------------------
        input: data=[Batch, input_feature_num]
        output: feature=[Batch, hdn_size]
        """
        self.fc1 = nn.Linear(input_feature_num, int(hdn_size/2)) 
        self.activation1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(int(hdn_size/2), int(hdn_size))
        self.activation2 = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.activation1(self.fc1(x))
        x = self.activation2(self.fc2(x))
        return x

class predictor(nn.Module):
    def __init__(self, embed_size, target_size):
        super(predictor, self).__init__()
        self.fc = nn.Linear(embed_size, target_size) 

        self.predict = nn.Sequential(
            nn.Linear(int(embed_size), int(embed_size/4)),
            nn.LeakyReLU(0.2),
            nn.Linear(int(embed_size/4), target_size),
        )

    def forward(self, feature):
        pred = self.predict(feature)
        return pred
    
class whole_model(nn.Module):
    def __init__(self, input_size, embed_size, target_size):
        super(whole_model, self).__init__()
        self.enc = encoder(input_size, embed_size)
        self.pred = predictor(embed_size, target_size)
    def forward(self, x):
        x = self.enc(x)
        output = self.pred(x)
        return output