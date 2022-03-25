import argparse
from config.parser import Parser


class BaseConfig(object):
    def __init__(self, config_path, profile_type=None):
        self.config = Parser.parse(config_path, profile_type)
        self.parser = argparse.ArgumentParser(description='GCNet.',
                                              formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                              conflict_handler="resolve")
        self._add_data()
        self._add_train()
        self._add_superparameter()

    def add_argument_group(self, argument: argparse.ArgumentParser):
        self.parser.add_argument_group(argument)
        return self.get_args()

    def get_args(self):
        args, _ = self.parser.parse_known_args()
        # print(args)
        return args

    def _add_data(self):
        param = self.config["data_path"]

        activate_index = param["setting"]["use_index"]
        img_path = param['path_list']['image_folder'][activate_index]
        mask_path = param['path_list']['mask_folder'][activate_index]
        model_path = param['path_list']['model_path'][activate_index]

        self.parser.add_argument('-im_p', '--img_path', dest='img_path', type=str,
                                 default=img_path, help='img path!')
        self.parser.add_argument('-ma_p', '--mask_path', dest='mask_path', type=str,
                                 default=mask_path, help='mask path!')
        self.parser.add_argument('-mo_p', '--model_path', dest='model_path', type=str,
                                 default=model_path, help='model path!')

    def _add_train(self):
        param = self.config["train"]
        param_model = param["model"]

        self.parser.add_argument('-p', '--pre_trained', dest='pre_trained', type=bool,
                                 default=param_model['pre_trained'], help='wheather pre-trained model')
        self.parser.add_argument('-p', '--continue_my_model', dest='continue_my_model', type=bool,
                                 default=param_model['continue_my_model'], help='wheather continue my trained model')
        self.parser.add_argument('-b', '--embedding_dim', type=int, default=param_model["embedding_dim"],
                                 help='embedding_dim', dest='embedding_dim')

    def _add_superparameter(self):
        self.parser.add_argument('-uhp', '--using_hard_pixel', action='store_true', help='weather using hard pixel',
                                 dest='using_hard_pixel')
        self.parser.add_argument('-ul', '--using_pseudo', action='store_true', help='weather using pseudo',
                                 dest='using_pseudo')
        self.parser.add_argument('-u', '--using_seed', action='store_true', help='weather using seed',
                                 dest='using_seed')
        self.parser.add_argument('-unn', '--uncertainty', action='store_true',
                                 help='weather using uncertainty threshold', dest='uncertainty')
        self.parser.add_argument('-unnn', '--negative_pseudo', action='store_true',
                                 help='weather using negative pseudo', dest='negative_pseudo')

        param = self.config["superparameter"]
        self.parser.add_argument('-sn', '--save_name', type=str, default=param['save_name'],
                                 help='the dir name you save', dest='save_name')

        static = param['static']
        self.parser.add_argument('-num_instance', '--num_instance', type=int, default=static['num_instance'],
                                 help='num_instance', dest='num_instance')
        self.parser.add_argument('-num_classes', '--num_classes', type=int, default=static['num_classes'],
                                 help='num_classes', dest='num_classes')
        self.parser.add_argument('-seed', '--seed', type=int, default=static['seed'],
                                 help='seed', dest='seed')
        self.parser.add_argument('-kf', '--k_fold', type=int, default=static['k_fold'],
                                 help='k_fold', dest='k_fold')
        self.parser.add_argument('-bs', '--batch_size', type=int, default=static['batch_size'],
                                 help='batch_size', dest='batch_size')

        train = param['train']
        self.parser.add_argument('-e', '--max_epoch', metavar='E', type=int, default=train["max_epoch"],
                                 help='Number of epochs', dest='max_epoch')
        self.parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?',
                                 default=train["learning_rate"],
                                 help='Learning rate', dest='learning_rate')
        self.parser.add_argument('-img_w', '--img_w', type=int, default=train['img_width'],
                                 help='img_w', dest='img_w')
        self.parser.add_argument('-img_h', '--img_h', type=int, default=train['img_height'],
                                 help='img_h', dest='img_h')

        net = param["networks"]
        self.parser.add_argument('--delta_v', default=net['delta_v'], type=float,
                                 help='the discriminativve loss parameter', dest='delta_v')
        self.parser.add_argument('--delta_d', default=net['delta_d'], type=float,
                                 help='the discriminativve loss parameter', dest='delta_d')
        self.parser.add_argument('-final_dim', '--final_dim', type=int, default=net['final_dim'],
                                 help='final_dim', dest='final_dim')

        loss = param["loss"]
        self.parser.add_argument('--p_cla', default=loss['loss_p']['p_cla'], type=float,
                                 help='the discriminativve loss parameter', dest='p_cla')
        self.parser.add_argument('--p_seg', default=loss['loss_p']['p_seg'], type=float,
                                 help='the discriminativve loss parameter', dest='p_seg')
        self.parser.add_argument('--p_disc', default=loss['loss_p']['p_disc'], type=float,
                                 help='the discriminativve loss parameter', dest='p_disc')

        self.parser.add_argument('--p_var', default=loss['disc_loss_p']['p_var'], type=float,
                                 help='the discriminativve loss parameter', dest='p_var')
        self.parser.add_argument('--p_dist', default=loss['disc_loss_p']['p_dist'], type=float,
                                 help='the discriminativve loss parameter', dest='p_dist')
        self.parser.add_argument('--p_reg', default=loss['disc_loss_p']['p_reg'], type=float,
                                 help='the discriminativve loss parameter', dest='p_reg')

        self.parser.add_argument('--version', action='version', version='%(prog)s 1.0')

        pse = param['pseudo']
        self.parser.add_argument('-e', '--start_pseudo_epoch', metavar='SE', type=int,
                                 default=pse["start_pseudo_epoch"], dest='start_pseudo_epoch',
                                 help='start using pseudo label when epoch=start_pseudo_epoch')

        self.parser.add_argument('-tau_n', '--tau_n', type=float, default=pse['tau_n'],
                                 help='tau_n', dest='tau_n')
        self.parser.add_argument('-kappa_n', '--kappa_n', type=float, default=pse['kappa_n'],
                                 help='kappa_n', dest='kappa_n')
        self.parser.add_argument('-tau_p', '--tau_p', type=float, default=pse['tau_p'],
                                 help='tau_p', dest='tau_p')
        self.parser.add_argument('-kappa_p', '--kappa_p', type=float, default=pse['kappa_p'],
                                 help='kappa_p', dest='kappa_p')
