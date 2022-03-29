from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--freeze_GB_encoder', type=bool, default=False, help="Freezes endoder from B to A (english encoder)")
        parser.add_argument('--on_colab', type=bool, default=False, help='runna su Colab o no')
        #parser.add_argument('--display_freq', type=int, default=64, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=64, help='frequency of showing training results on console')

        # network saving and loading parameters
        parser.add_argument('--eval_freq', type=int, default=16384, help='frequency of saving the latest results')
        parser.add_argument('--save_latest_freq', type=int, default=512, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1,
                            help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--iter_count', type=int, default=0,
                            help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')

        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        #parser.add_argument('--lr', type=float, default=0.00001, help='initial learning rate for adam')
        parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='crossentropy', help='the type of GAN objective. [crossentropy | lsgan]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')


        parser.add_argument('--train_percentage', type=float, default=0.8, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--eval_percentage', type=float, default=0.1, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--test_percentage', type=float, default=0.1, help='multiply by a gamma every lr_decay_iters iterations')
        self.isTrain = True
        return parser
