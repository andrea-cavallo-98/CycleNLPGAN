import logging
from copy import deepcopy

import sklearn
from sklearn.metrics import pairwise_distances
import torch
import itertools

import sacrebleu

from losses import CosineSimilarityLoss, MSELoss
from .base_model import BaseModel
from . import networks
import numpy as np
import gc

class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.add_argument('--task', type=str, default='reconstruction',
                            help='specify the task of the CycleGAN [translation|reconstruction]')
        parser.add_argument('--network_type', type=str, default='marianMT',
                            help='specify generator architecture and language [marianMT|t5]')

        parser.add_argument('--language', type=str, default='vi', help='specify destination language')
        parser.add_argument('--init-language', type=str, default='zh', help='specify destination language of original model (en-XX)')

        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--loss-type', type=str, default="mse", help='Loss used in cycle [mse|cosine]')

            parser.add_argument('--lambda_G', type=float, default=10.0, help='scaling factor for generator loss')
            parser.add_argument('--lambda_D', type=float, default=10.0, help='scaling factor for discriminator loss')
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_A_sup', type=float, default=10.0, help='weight for supervised loss')
            parser.add_argument('--lambda_B_sup', type=float, default=10.0, help='weight for supervised loss')
           
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_AB', 'G_AB', 'cycle_ABA', 'D_BA', 'G_BA', 'cycle_BAB']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_AB', 'G_BA', 'D_AB', 'D_BA']
        else:  # during test time, only load Gs
            self.model_names = ['G_AB', 'G_BA']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        self.netG_AB, self.netG_BA = networks.define_Gs(opt.task, opt.network_type, 'en', opt.language, opt.init_language, self.gpu_ids, opt.freeze_GB_encoder)

        if self.isTrain:  # define discriminators

            #netDAB_name = networks.define_name(opt.netD, 'en')
            #netDBA_name = networks.define_name(opt.netD, opt.language)
            netDAB_name = "distilbert-base-multilingual-cased"
            netDBA_name = "distilbert-base-multilingual-cased"

            self.netD_AB = networks.define_D(opt.netD, netDAB_name, self.gpu_ids)
            self.netD_BA = networks.define_D(opt.netD, netDBA_name, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.

            if opt.loss_type == 'cosine':
                self.criterionCycle = CosineSimilarityLoss().to(self.device)
            elif opt.loss_type == 'mse':
                self.criterionCycle = MSELoss().to(self.device)
            else:
                raise NotImplementedError(opt.loss_type + " not implemented")

            self.criterionIdt = torch.nn.CosineEmbeddingLoss()  # CosineSimilarityLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            if opt.freeze_GB_encoder is False:
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_AB.parameters(), self.netG_BA.parameters()),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_AB.parameters(), self.netG_BA.module.model.base_model.decoder.parameters()),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_AB.parameters(), self.netD_BA.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        self.loss_G_AB = 0
        self.loss_G_BA = 0
        self.loss_D_AB = 0
        self.loss_D_BA = 0
        self.loss_G_AB_sup = 0
        self.loss_G_BA_sup = 0
        self.loss_cycle_ABA = 0
        self.loss_cycle_BAB = 0
        self.loss_G = 0


    def set_input(self, input_A, input_B):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        # torch.cuda.empty_cache()
        self.real_A = input_A
        self.real_B = input_B


    def forward(self, supervised=False):

        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        if supervised:
            self.fake_B, self.loss_G_AB_sup = self.netG_AB(self.real_A, self.real_B)  # G_A(A)
            ### Add self.real_A as labels for second pass
            self.rec_A, self.loss_cycle_ABA = self.netG_BA(self.fake_B, self.real_A)  # G_B(G_A(A))

            self.fake_A, self.loss_G_BA_sup = self.netG_BA(self.real_B, self.real_A)  # G_B(B)
            self.rec_B, self.loss_cycle_BAB = self.netG_AB(self.fake_A, self.real_B)  # G_A(G_B(B))

        else:    
            self.fake_B = self.netG_AB(self.real_A, None)  # G_A(A)
            ### Add self.real_A as labels for second pass
            self.rec_A, self.loss_cycle_ABA = self.netG_BA(self.fake_B, self.real_A)  # G_B(G_A(A))

            self.fake_A = self.netG_BA(self.real_B, None)  # G_B(B)
            self.rec_B, self.loss_cycle_BAB = self.netG_AB(self.fake_A, self.real_B)  # G_A(G_B(B))


    def backward_D_basic(self, netD, real_sent, fake_sent):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        #pred_real = netD(real)
        loss_D_real = netD(real_sent, 1).loss
        # Fake
        loss_D_fake = netD(fake_sent, 0).loss
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5 * self.opt.lambda_D

        loss_D.backward()
        return loss_D#.item()


    def backward_D_AB(self):
        """Calculate GAN loss for discriminator D_A"""
        self.loss_D_AB = self.backward_D_basic(self.netD_AB, self.real_B, self.fake_B)
        self.loss_D_AB = self.loss_D_AB.item()

    def backward_D_BA(self):
        """Calculate GAN loss for discriminator D_B"""
        self.loss_D_BA = self.backward_D_basic(self.netD_BA, self.real_A, self.fake_A)
        self.loss_D_BA = self.loss_D_BA.item()

    def backward_G(self, supervised=False):
        """Calculate the loss for generators G_A and G_B"""

        lambda_G = self.opt.lambda_G
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_A_sup = self.opt.lambda_A_sup
        lambda_B_sup = self.opt.lambda_B_sup

        # Loss G_AB is the one of the discriminator
        self.loss_G_AB = self.netD_AB(self.fake_B, 1).loss
        self.loss_G_AB = self.loss_G_AB * 0.5 * lambda_G

        self.loss_G_BA = self.netD_BA(self.fake_A, 1).loss
        self.loss_G_BA = self.loss_G_BA * 0.5 * lambda_G

        if supervised:
            # combined loss and calculate gradients
            self.loss_G = self.loss_G_AB + self.loss_G_BA + self.loss_cycle_ABA * lambda_A + self.loss_cycle_BAB * lambda_B + \
                            self.loss_G_AB_sup * lambda_A_sup + self.loss_G_BA_sup * lambda_B_sup
        else:
            # combined loss and calculate gradients
            self.loss_G = self.loss_G_AB + self.loss_G_BA + self.loss_cycle_ABA * lambda_A + self.loss_cycle_BAB * lambda_B  

        self.loss_G.backward()

        torch.cuda.empty_cache()
        gc.collect()



    def optimize_parameters(self, supervised = False):
        """Calculate losses, gradients, and update network weights; called in every training iteration
            set supervised parameter to choose between supervised and unsupervised training
        """

        self.netG_AB.train()
        self.netG_BA.train()
        self.netD_AB.train()
        self.netD_BA.train()

        # forward
        self.forward(supervised)  
        gc.collect()

        self.set_requires_grad([self.netD_AB, self.netD_BA], False)
        torch.enable_grad()

        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G(supervised)  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights

        if supervised:
            self.loss_G_AB_sup = self.loss_G_AB_sup.item()
            self.loss_G_BA_sup = self.loss_G_BA_sup.item()
            #del self.loss_G_AB_sup
            #del self.loss_G_BA_sup
        
        self.loss_G_AB = self.loss_G_AB.item()
        self.loss_G_BA = self.loss_G_BA.item()
        self.loss_G = self.loss_G.item()
        #del self.loss_G_AB
        #del self.loss_G_BA
        #del self.loss_G
        gc.collect()
        
        # D_A and D_B
        self.set_requires_grad([self.netD_AB], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_AB()  # calculate gradients for D_A

        self.set_requires_grad([self.netD_BA], True)
        self.backward_D_BA()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

        del self.fake_A
        del self.fake_B
        del self.rec_A
        del self.rec_B
        torch.no_grad()
        torch.cuda.empty_cache()
        gc.collect()


    def evaluate(self, sentences_file="eval_sentences.txt", sacre_file="sacre_bleu.tsv"):
        #logging.info("\n\nEvaluating...")

        self.netG_AB.module.eval()
        self.netG_BA.module.eval()
        self.netD_AB.module.eval()
        self.netD_BA.module.eval()
        
        with torch.no_grad():
            self.forward()  # calculate loss functions, get gradients, update network weights
        gc.collect()
        with open(sentences_file, "a") as sentences_file:
            for j in range(len(self.real_A)):
                str1 = " A->B->A : " + self.real_A[j] + " -> " + self.fake_B[j] + " -> " + self.rec_A[j]
                str2 = " B->A->B : " + self.real_B[j] + " -> " + self.fake_A[j] + " -> " + self.rec_B[j]
                #logging.info(str1)
                #logging.info(str2)
                sentences_file.write('%s\n' % str1)  # save the message
                sentences_file.write('%s\n\n' % str2)  # save the message


        bleu_fake_A = sacrebleu.raw_corpus_bleu(self.fake_A, [self.real_A]).score
        bleu_rec_A = sacrebleu.raw_corpus_bleu(self.rec_A, [self.real_A]).score
        bleu_fake_B = sacrebleu.raw_corpus_bleu(self.fake_B, [self.real_B]).score
        bleu_rec_B = sacrebleu.raw_corpus_bleu(self.rec_B, [self.real_B]).score

        with open(sacre_file, "a") as sacre_file:
                sacre_file.write(str(bleu_fake_A) + '\t' + str(bleu_rec_A) + '\t' +str(bleu_fake_B) + '\t' +str(bleu_rec_B) + '\n')

        self.netG_AB.module.train()
        self.netG_BA.module.train()
        self.netD_AB.module.train()
        self.netD_BA.module.train()
        gc.collect()
        
        
    def evaluate_unsupervised(self, sentences_file="eval_sentences.txt"):
        #logging.info("\n\nEvaluating...")

        self.netG_AB.module.eval()
        self.netG_BA.module.eval()
        self.netD_AB.module.eval()
        self.netD_BA.module.eval()
        
        with torch.no_grad():
            self.forward()  # calculate loss functions, get gradients, update network weights
        gc.collect()
        with open(sentences_file, "a") as sentences_file:
            for j in range(len(self.real_A)):
                str1 = " A->B->A : " + self.real_A[j] + " -> " + self.fake_B[j] + " -> " + self.rec_A[j]
                str2 = " B->A->B : " + self.real_B[j] + " -> " + self.fake_A[j] + " -> " + self.rec_B[j]
                #logging.info(str1)
                #logging.info(str2)
                sentences_file.write('%s\n' % str1)  # save the message
                sentences_file.write('%s\n\n' % str2)  # save the message

        '''
        bleu_fake_A = sacrebleu.raw_corpus_bleu(self.fake_A, [self.real_A]).score
        bleu_rec_A = sacrebleu.raw_corpus_bleu(self.rec_A, [self.real_A]).score
        bleu_fake_B = sacrebleu.raw_corpus_bleu(self.fake_B, [self.real_B]).score
        bleu_rec_B = sacrebleu.raw_corpus_bleu(self.rec_B, [self.real_B]).score

        with open(sacre_file, "a") as sacre_file:
                sacre_file.write(str(bleu_fake_A) + '\t' + str(bleu_rec_A) + '\t' +str(bleu_fake_B) + '\t' +str(bleu_rec_B) + '\n')
        '''
        self.netG_AB.module.train()
        self.netG_BA.module.train()
        self.netD_AB.module.train()
        self.netD_BA.module.train()
        gc.collect()



    def load_networks(self, epoch):
        BaseModel.load_networks(self, epoch)
        if self.isTrain:
            self.optimizers = []
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            if self.opt.freeze_GB_encoder is False:
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_AB.parameters(), self.netG_BA.parameters()),
                                                    lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            else:
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_AB.parameters(), self.netG_BA.module.model.base_model.decoder.parameters()),
                                                    lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_AB.module.parameters(), self.netD_BA.module.parameters()),
                                                lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
