from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from transformers import EncoderDecoderModel, MarianTokenizer, MarianMTModel

from . import EncDecModel,Pooling
from .EncDecT5Model import EncDecT5Model
from .discriminator_transformer import DiscriminatorTransformer


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_net(net, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net = net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)

    return net

def define_name(model_name,language):
    if model_name == 'mpl':
        return model_name
    elif model_name == 'bert-base':
        complete_name = model_name
        if language == 'en':
            complete_name = complete_name
        elif language == 'de':
            complete_name = complete_name + "-german"
        elif language == 'it':
            complete_name = "dbmdz/bert-base-italian"
        else:
            raise NotImplementedError('Language [%s] is not implemented', language)

        complete_name = complete_name + "-cased"
    elif model_name == 'distilbert-base':
        complete_name = model_name
        if language == 'en':
            complete_name = complete_name
        else:
            complete_name = complete_name + "-multilingual"

        complete_name = complete_name + "-cased"

    return complete_name


def define_language(language):
    ret_val = ""
    if language == "en":
        ret_val = "English"
    elif language == "it":
        ret_val = "Italian"
    elif language == "es":
        ret_val = "Spanish"
    elif language == "de":
        ret_val = "German"
    elif language == "fr":
        ret_val = "French"
    elif language == "zh":
        ret_val = "Chinese"
    elif language == "ru":
        ret_val = "Russian"
    else:
        raise NotImplementedError("Language not supported")

    return ret_val


def define_Gs(task, net_type, source, dest, init_dest, gpu_ids=[], freeze_GB_encoder=False):

    netA = define_G(net_type, source, dest, init_dest, gpu_ids, use_init_net=False, freeze_encoder=False)
    netB = define_G(net_type, dest, source, init_dest, gpu_ids, use_init_net=False, freeze_encoder=freeze_GB_encoder)
    netA.model.base_model.train()
    netB.model.base_model.train()

    if freeze_GB_encoder is True:
        netB.model.base_model.encoder.eval()

    netA.task = task
    netB.task = task
    
    netA = init_net(netA, gpu_ids)
    netB = init_net(netB, gpu_ids)

    return netA, netB


def define_G(model, source, dest, init_dest, gpu_ids=[], use_init_net= True, freeze_encoder=False):
    """Create a generator

    Parameters:
        model (str) -- the type of the network: encoder | encoder-decoder
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None


    if model == 't5':
        src_lang = define_language(source)
        tgt_lang = define_language(dest)
        model_name = 't5-small'
        net = EncDecT5Model(model_name, freeze_encoder=freeze_encoder, source_language=src_lang, target_language=tgt_lang)

    elif model == 'marianMT':
        model_name = 'Helsinki-NLP/opus-mt-'+source+'-'+init_dest
        net = EncDecModel(model_name, freeze_encoder=freeze_encoder, source_lang=source, target_lang=dest)
    
    else:
        net = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-cased', 'bert-base-german-cased')    #net=SentenceTransformer(netG)

    if use_init_net == True:
        return init_net(net, gpu_ids)
    else:
        return net


def define_D(netD, netD_name, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None

    if netD == 'distilbert-base':
        net = DiscriminatorTransformer(netD_name)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'crossentropy':
            self.loss = nn.BCELoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'crossentropy':
            target_tensor = self.get_target_tensor(prediction, target_is_real)#.type(torch.LongTensor)
            loss = self.loss(prediction.type(torch.float32), target_tensor)
            loss = loss/100.0 #done because BCELoss returns a value between 0 and 100
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
