import torch
import torch.nn as nn
import random
import copy
import torch.nn.functional as F

class ResidualAE(nn.Module):
    ''' Residual autoencoder using fc layers
        layers should be something like [128, 64, 32]
        eg:[128,64,32]-> add: [(input_dim, 128), (128, 64), (64, 32), (32, 64), (64, 128), (128, input_dim)]
                          concat: [(input_dim, 128), (128, 64), (64, 32), (32, 64), (128, 128), (256, input_dim)]
    '''
    def __init__(self, layers, n_blocks, input_dim, dropout=0.5, use_bn=False):
        super(ResidualAE, self).__init__()
        self.use_bn = use_bn
        self.dropout = dropout
        self.n_blocks = n_blocks
        self.input_dim = input_dim
        self.transition = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )
        for i in range(n_blocks):
            setattr(self, 'encoder_' + str(i), self.get_encoder(layers))
            setattr(self, 'decoder_' + str(i), self.get_decoder(layers))
    
    def get_encoder(self, layers):
        all_layers = []
        input_dim = self.input_dim
        for i in range(0, len(layers)):
            all_layers.append(nn.Linear(input_dim, layers[i]))
            all_layers.append(nn.LeakyReLU())
            if self.use_bn:
                all_layers.append(nn.BatchNorm1d(layers[i]))
            if self.dropout > 0:
                all_layers.append(nn.Dropout(self.dropout))
            input_dim = layers[i]
        # delete the activation layer of the last layer
        decline_num = 1 + int(self.use_bn) + int(self.dropout > 0)
        all_layers = all_layers[:-decline_num]
        return nn.Sequential(*all_layers)
    
    def get_decoder(self, layers):
        all_layers = []
        decoder_layer = copy.deepcopy(layers)
        decoder_layer.reverse()
        decoder_layer.append(self.input_dim)
        for i in range(0, len(decoder_layer)-2):
            all_layers.append(nn.Linear(decoder_layer[i], decoder_layer[i+1]))
            all_layers.append(nn.ReLU()) # LeakyReLU
            if self.use_bn:
                all_layers.append(nn.BatchNorm1d(decoder_layer[i]))
            if self.dropout > 0:
                all_layers.append(nn.Dropout(self.dropout))
        
        all_layers.append(nn.Linear(decoder_layer[-2], decoder_layer[-1]))
        return nn.Sequential(*all_layers)

    
    def forward(self, x):
        x_in = x
        x_out = x.clone().fill_(0)
        latents = []
        for i in range(self.n_blocks):
            encoder = getattr(self, 'encoder_' + str(i))
            decoder = getattr(self, 'decoder_' + str(i))
            x_in = x_in + x_out
            latent = encoder(x_in)
            x_out = decoder(latent)
            latents.append(latent)
        latents = torch.cat(latents, dim=-1)
        return self.transition(x_in+x_out), latents

class ResidualUnetAE(nn.Module):
    ''' Residual autoencoder using fc layers
    '''
    def __init__(self, layers, n_blocks, input_dim, dropout=0.5, use_bn=False, fusion='concat'):
        ''' Unet是对称的, 所以layers只用写一半就好 
            eg:[128,64,32]-> add: [(input_dim, 128), (128, 64), (64, 32), (32, 64), (64, 128), (128, input_dim)]
                          concat: [(input_dim, 128), (128, 64), (64, 32), (32, 64), (128, 128), (256, input_dim)]
        '''
        super(ResidualUnetAE, self).__init__()
        self.use_bn = use_bn
        self.dropout = dropout
        self.n_blocks = n_blocks
        self.input_dim = input_dim
        self.layers = layers
        self.fusion = fusion
        if self.fusion == 'concat':
            self.expand_num = 2
        elif self.fusion == 'add':
            self.expand_num = 1
        else:
            raise NotImplementedError('Only concat and add is available')
    
        # self.encoder = self.get_encoder(layers)
        for i in range(self.n_blocks):
            setattr(self, 'encoder_'+str(i), self.get_encoder(layers))
            setattr(self, 'decoder_'+str(i), self.get_decoder(layers))

    
    def get_encoder(self, layers):
        encoder = []
        input_dim = self.input_dim
        for i in range(0, len(layers)):
            layer = []
            layer.append(nn.Linear(input_dim, layers[i]))
            layer.append(nn.LeakyReLU())
            if self.use_bn:
                layer.append(nn.BatchNorm1d(layers[i]))
            if self.dropout > 0:
                layer.append(nn.Dropout(self.dropout))
            layer = nn.Sequential(*layer)
            encoder.append(layer)
            input_dim = layers[i]
        encoder = nn.Sequential(*encoder)
        return encoder
    
    def get_decoder(self, layers):
        decoder = []
        # first layer don't need to fusion outputs
        first_layer = []
        first_layer.append(nn.Linear(layers[-1], layers[-2]))
        if self.use_bn:
            first_layer.append(nn.BatchNorm1d(layers[-1] * self.expand_num))
        if self.dropout > 0:
            first_layer.append(nn.Dropout(self.dropout))
        decoder.append(nn.Sequential(*first_layer))
    
        for i in range(len(layers)-2, 0, -1):
            layer = []
            layer.append(nn.Linear(layers[i]*self.expand_num, layers[i-1]))
            layer.append(nn.LeakyReLU())
            if self.use_bn:
                layer.append(nn.BatchNorm1d(layers[i] * self.expand_num))
            if self.dropout > 0:
                layer.append(nn.Dropout(self.dropout))
            layer = nn.Sequential(*layer)
            decoder.append(layer)
        
        decoder.append(
            nn.Sequential(
                nn.Linear(layers[0] * self.expand_num, self.input_dim),
                nn.ReLU()
            )
        )
        decoder = nn.Sequential(*decoder)
        return decoder
    
    def forward_AE_block(self, x, block_num):
        encoder = getattr(self, 'encoder_' + str(block_num))
        decoder = getattr(self, 'decoder_' + str(block_num))
        encoder_out_lookup = {}
        x_in = x
        for i in range(len(self.layers)):
            x_out = encoder[i](x_in)
            encoder_out_lookup[i] = x_out.clone()
            x_in = x_out
        
        for i in range(len(self.layers)):
            encoder_out_num = len(self.layers) -1 - i
            encoder_out = encoder_out_lookup[encoder_out_num]
            if i == 0:
                pass
            elif self.fusion == 'concat':
                x_in = torch.cat([x_in, encoder_out], dim=-1)
            elif self.fusion == 'add':
                x_in = x_in + encoder_out
            
            x_out = decoder[i](x_in)
            x_in = x_out
        
        return x_out
    
    def forward(self, x):
        x_in = x
        x_out = x.clone().fill_(0)
        output = {}
        for i in range(self.n_blocks):
            x_in = x_in + x_out
            x_out = self.forward_AE_block(x_in, i)
            output[i] = x_out.clone()

        return x_out, output

class SimpleFcAE(nn.Module):
    def __init__(self, layers, input_dim, dropout=0.5, use_bn=False):
        ''' Parameters:
            --------------------------
            input_dim: input feature dim
            layers: [x1, x2, x3] will create 3 layers with x1, x2, x3 hidden nodes respectively.
            dropout: dropout rate
            use_bn: use batchnorm or not
        '''
        super().__init__()
        self.input_dim = input_dim
        self.dropout = dropout
        self.use_bn = use_bn
        self.encoder = self.get_encoder(layers)
        self.decoder = self.get_decoder(layers)
        
    def get_encoder(self, layers):
        all_layers = []
        input_dim = self.input_dim
        for i in range(0, len(layers)):
            all_layers.append(nn.Linear(input_dim, layers[i]))
            all_layers.append(nn.LeakyReLU())
            if self.use_bn:
                all_layers.append(nn.BatchNorm1d(layers[i]))
            if self.dropout > 0:
                all_layers.append(nn.Dropout(self.dropout))
            input_dim = layers[i]
        return nn.Sequential(*all_layers)
    
    def get_decoder(self, layers):
        all_layers = []
        decoder_layer = copy.deepcopy(layers)
        decoder_layer.reverse()
        decoder_layer.append(self.input_dim)
        for i in range(0, len(decoder_layer)-1):
            all_layers.append(nn.Linear(decoder_layer[i], decoder_layer[i+1]))
            all_layers.append(nn.ReLU()) if i == len(decoder_layer)-2 else all_layers.append(nn.LeakyReLU()) 
            if self.use_bn:
                all_layers.append(nn.BatchNorm1d(decoder_layer[i]))
            if self.dropout > 0:
                all_layers.append(nn.Dropout(self.dropout))
        
        # all_layers.append(nn.Linear(decoder_layer[-2], decoder_layer[-1]))
        return nn.Sequential(*all_layers)
    
    def forward(self, x):
        ## make layers to a whole module
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent
