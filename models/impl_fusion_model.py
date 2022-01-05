
import torch
import os
import json
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.fc import FcEncoder
from models.networks.lstm import LSTMEncoder
from models.networks.textcnn import TextCNN
from models.networks.classifier import FcClassifier

''' Implementation of 
    IMPLICIT FUSION BY JOINT AUDIOVISUAL TRAINING FOR EMOTION RECOGNITION IN MONO MODALITY
    https://ieeexplore.ieee.org/document/8682773
'''

class ImplFusionModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--input_dim_a', type=int, default=130, help='acoustic input dim')
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=384, help='visual input dim')
        parser.add_argument('--weight_a', type=float, default=1.0, help='audio loss weight')
        parser.add_argument('--weight_v', type=float, default=0.3, help='audio loss weight')
        parser.add_argument('--weight_l', type=float, default=0.3, help='audio loss weight')
        parser.add_argument('--embd_size', default=128, type=int, help='embedding size for each modality')
        parser.add_argument('--embd_method_a', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
            help='audio embedding method,last,mean or atten')
        parser.add_argument('--embd_method_v', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
            help='visual embedding method,last,mean or atten')
        parser.add_argument('--cls_layers', type=str, default='128,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='rate of dropout')
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')
        parser.add_argument('--trn_modality', type=str, help='which modality used for training for model')
        parser.add_argument('--test_modality', type=str, help='which modality used for testing for model')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        # our expriment is on 10 fold setting, teacher is on 5 fold setting, the train set should match
        self.loss_names = []
        self.model_names = ['C']
        self.trn_modality = opt.trn_modality
        self.test_modality = opt.test_modality
        assert len(self.test_modality) == 1
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        self.netC = FcClassifier(opt.embd_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)
        
        # acoustic model
        if 'A' in self.trn_modality:
            self.model_names.append('A')
            self.loss_names.append('CE_A')
            self.netA = LSTMEncoder(opt.input_dim_a, opt.embd_size, embd_method=opt.embd_method_a)
            self.weight_a = opt.weight_a
            
        # lexical model
        if 'L' in self.trn_modality:
            self.model_names.append('L')
            self.loss_names.append('CE_L')
            self.netL = TextCNN(opt.input_dim_l, opt.embd_size)
            self.weight_l = opt.weight_l
            
        # visual model
        if 'V' in self.trn_modality:
            self.model_names.append('V')
            self.loss_names.append('CE_V')
            self.netV = LSTMEncoder(opt.input_dim_v, opt.embd_size, opt.embd_method_v)
            self.weight_v = opt.weight_v
            
        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.output_dim = opt.output_dim

        # modify save_dir
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
    
    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        if 'A' in self.trn_modality:
            self.acoustic = input['A_feat'].float().to(self.device)
        if 'L' in self.trn_modality:
            self.lexical = input['L_feat'].float().to(self.device)
        if 'V' in self.trn_modality:
            self.visual = input['V_feat'].float().to(self.device)
        
        self.label = input['label'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        modality = self.trn_modality if self.isTrain else self.test_modality
        if 'A' in modality:
            self.feat_A = self.netA(self.acoustic)
            self.logits_A, _ = self.netC(self.feat_A)
            self.pred = F.softmax(self.logits_A, dim=-1)

        if 'L' in modality:
            self.feat_L = self.netL(self.lexical)
            self.logits_L, _ = self.netC(self.feat_L)
            self.pred = F.softmax(self.logits_L, dim=-1)
        
        if 'V' in modality:
            self.feat_V = self.netV(self.visual)
            self.logits_V, _ = self.netC(self.feat_V)
            self.pred = F.softmax(self.logits_V, dim=-1)
        
    def backward(self):
        """Calculate the loss for back propagation"""
        losses = []
        if 'A' in self.trn_modality:
            self.loss_CE_A = self.criterion_ce(self.logits_A, self.label) * self.weight_a
            losses.append(self.loss_CE_A)

        if 'L' in self.trn_modality:
            self.loss_CE_L = self.criterion_ce(self.logits_L, self.label) * self.weight_l
            losses.append(self.loss_CE_L)
            
        if 'V' in self.trn_modality:
            self.loss_CE_V = self.criterion_ce(self.logits_V, self.label) * self.weight_v
            losses.append(self.loss_CE_L)
            
        loss = sum(losses)
        loss.backward()
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 0.5)

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()   
        # backward
        self.optimizer.zero_grad()  
        self.backward()            
        self.optimizer.step() 
