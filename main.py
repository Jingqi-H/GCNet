import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
# from thop import profile, clever_format
from torch.utils.data import DataLoader
import numpy as np
import time
import datetime
import json
import random
import matplotlib.pyplot as plt
from torchvision import transforms

from dataset.MyImageFolder import Modified_Metric_Learning_ImageFolder, StandarImageFolder, k_fold_loader
from utils.wheels import save_intermediate, save_best, save_k_final_results, init_dir, s2t
from utils.visulize import plot_loss_acc
from utils.metrics import metrics_score_multi
from loss.discriminative_loss import DiscriminativeLoss_klintan
from loss.dice_loss import MultiDiceLoss
from trainer import train
# from models.resnet_unet_metric import ResNetUNetMetric
from models.gcnet import GCNet
from eval import val_test, test
from config.config import BaseConfig
from dataset.augumentation import AddPepperNoise


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print('set seed:', seed, '!')


def main(parser):
    args = parser.get_args()
    print(args)
    if args.using_pseudo:
        print('use pseudo.')
    else:
        print('not use pseudo.')
    if args.using_seed:
        setup_seed(args.seed)
    save_dir, save_intermediate_dir, save_display_dir, save_csv_dir, save_best_dir, save_pre_img = init_dir(
        args.save_name)

    transform = transforms.Compose([
        AddPepperNoise(snr=0.95, p=0.5),
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = Modified_Metric_Learning_ImageFolder(root=args.img_path + '/data_train', mask_root=args.mask_path,
                                                         transform=transform)

    # {'narrow1':0, 'narrow2':1, 'narrow3':2, 'narrow4':3, 'wide':4}
    narrow_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in narrow_list.items())
    cla = []
    for key, val in narrow_list.items():
        cla.append(key)
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    indices = list(range(len(train_dataset)))
    print(len(train_dataset))
    np.random.shuffle(indices)

    for i in range(args.k_fold):
        print('\n', '*' + '-' * 10, 'F{}'.format(i + 1), '-' * 10 + '*')

        # train_result = pd.DataFrame(columns=('loss', 'accurate'))
        # val_result = pd.DataFrame(columns=('loss', 'accurate', 'recall', 'precision', 'AUC', 'F1'))
        # test_result = pd.DataFrame(columns=('loss', 'accurate', 'recall', 'precision', 'AUC', 'F1'))

        train_len, train_loader, validation_loader = k_fold_loader(i, int(len(train_dataset) * 1 // args.k_fold),
                                                                   indices,
                                                                   train_dataset, args.batch_size)
        test_data = StandarImageFolder(root=os.path.join(args.img_path, 'data_test'),
                                       transform=transforms.Compose([
                                           transforms.Resize((args.img_h, args.img_w)),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
                                       )
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=16)

        disc_criterion = DiscriminativeLoss_klintan(parser).cuda()
        cla_criterion = nn.CrossEntropyLoss().cuda()
        seg_criterion = MultiDiceLoss(num_classes=args.num_classes)

        net = GCNet(parser)
        net = net.cuda()
        print(net)
        for param in net.extractors.conv1.parameters():
            param.requires_grad = False
        for param in net.extractors.bn1.parameters():
            param.requires_grad = False
        for param in net.extractors.layer1.parameters():
            param.requires_grad = False
        for p in net.extractors.layer2.parameters():
            p.requires_grad = False

        # parm = {}
        # for name, parameters in net.named_parameters():
        #     if name == 'extractors.conv1.weight':
        #         print(name, ':', parameters.size())
        #         print(name, ':', parameters)
        #         parm[name] = parameters.cpu().detach().numpy()

        for name, param in net.named_parameters():
            if param.requires_grad:
                print(name)
        # flops, params = profile(net, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
        # flops, params = clever_format([flops, params])
        # print('# Model Params: {} FLOPs: {}'.format(params, flops))

        optimizer = optim.SGD(net.parameters(),
                              lr=args.learning_rate,
                              momentum=0.9,
                              dampening=0,
                              weight_decay=0.0005,
                              nesterov=True, )

        # results = {'train_loss': [], 'train_acc@1': [], 'train_acc@2': [],
        #            'val_loss': [], 'val_acc@1': [], 'val_acc@2': [],
        #            'test_loss': [], 'test_acc@1': [], 'test_acc@2': [], 'test_auc': [], 'test_f1': []}
        results = {'train_loss': [], 'train_acc@1': [], 'train_acc@2': [],
                   'val_loss': [], 'val_acc@1': [], 'val_acc@2': [],}
        train_part_loss = {'total_loss': [], 'cla_loss': [], 'seg_loss': [], 'em_loss': []}

        best_acc, best_recall, best_precision, best_auc, best_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
        lr_epoch = []
        for epoch in range(1, args.max_epoch + 1):
            print('\nF{} | Epoch [{}/{}]'.format(i + 1, epoch, args.max_epoch))

            # 1 train
            train_dict = train(args,
                               per_epoch=epoch,
                               net=net,
                               train_dataset=train_dataset,
                               data_loader=train_loader,
                               train_optimizer=optimizer,
                               disc_loss_func=disc_criterion,
                               seg_loss_func=seg_criterion,
                               cla_loss_func=cla_criterion)
            train_part_loss['total_loss'].append(train_dict['loss'])
            train_part_loss['cla_loss'].append(train_dict['loss_cla'])
            train_part_loss['seg_loss'].append(train_dict['loss_seg'])
            train_part_loss['em_loss'].append(train_dict['loss_em'])

            lr_epoch += train_dict['lr']
            # print('lr:', lr)

            results['train_loss'].append(train_dict['loss'])
            results['train_acc@1'].append(train_dict['acc_1'].item())
            results['train_acc@2'].append(train_dict['acc_2'].item())

            val_dict = val_test(per_epoch=epoch,
                                pseudo=[args.using_pseudo, args.tau_p, args.start_pseudo_epoch],
                                val_dataset=train_dataset,
                                net=net,
                                data_loader=validation_loader,
                                disc_loss=disc_criterion,
                                seg_loss=seg_criterion,
                                cla_loss=cla_criterion,
                                p_seg=args.p_seg,
                                p_discriminative=args.p_disc,
                                p_cla=args.p_cla,
                                save_pre=save_pre_img,
                                is_val=True)
            results['val_loss'].append(val_dict['val_loss'].item())
            results['val_acc@1'].append(val_dict['val_acc_1'].item())
            results['val_acc@2'].append(val_dict['val_acc_2'].item())

            val_acc, val_recall, val_precision, val_auc, val_f1 = metrics_score_multi(val_dict['val_gt_labels'],
                                                                                val_dict['val_pred_probs'])

            # 3 test
            # test_loss, test_acc_1, test_acc_2, test_pred_probs, test_pred_labels, test_gt_labels = test(net,
            #                                                                                             test_loader,
            #                                                                                             cla_criterion)
            #
            # results['test_loss'].append(test_loss)
            # results['test_acc@1'].append(test_acc_1)
            # results['test_acc@2'].append(test_acc_2)
            #
            # test_acc, test_recall, test_precision, test_auc, test_f1 = metrics_score_multi(test_gt_labels, test_pred_probs)
            # results['test_auc'].append(test_auc)
            # results['test_f1'].append(test_f1)

            '''save statistics'''
            data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
            data_frame.to_csv(os.path.join(save_csv_dir, 'final_statistics_' + 'K' + str(i + 1) + '.csv'),
                              index_label='epoch')
            total_curve = plot_loss_acc(data_frame)
            total_curve.savefig(os.path.join(save_dir, 'final_statistics_' + 'K' + str(i + 1) + '.png'))

            part_loss_frame = pd.DataFrame(data=train_part_loss)
            part_loss_frame.to_csv(os.path.join(save_csv_dir, 'train_part_loss_' + 'K' + str(i + 1) + '.csv'),
                                   index=False)
            plt.cla()
            plt.close('all')
            part_loss_frame.plot()
            plt.savefig(os.path.join(save_dir, 'train_part_loss_' + 'K' + str(i + 1) + '.png'))

            torch.save(net.state_dict(),
                       os.path.join(save_dir, 'FinalK' + str(i + 1) + '.pth'))
            if val_dict['val_acc_1'] > best_acc:
                best_acc = val_dict['val_acc_1']
                save_best(i, net, val_dict['val_gt_labels'], val_dict['val_pred_labels'],
                          val_dict['val_pred_probs'], cla, save_best_dir, save_dir)
                # save_best(i, net, val_dict['val_gt_labels'], val_dict['val_pred_labels'],
                #           val_dict['val_pred_probs'], test_gt_labels, test_pred_labels, cla,
                #           test_pred_probs, save_best_dir, save_dir)

                print('[Best]\nVal:  acc:{} | recalll:{} | precision:{} | auc:{} | f1:{}'.format(val_acc, val_recall,
                                                                                                 val_precision, val_auc,
                                                                                                 val_f1))
                print('each iou: {}\n{}'.format(val_dict['five_iou'].shape, val_dict['five_iou']))
                # print('Test: acc:{} | auc:{} | f1:{}'.format(test_acc, test_auc, test_f1))

            # if epoch % 100 == 0:
            #     save_intermediate(net, save_intermediate_dir, epoch,
            #                       i, val_gt_labels, val_pred_labels,
            #                       val_pred_probs, cla, test_pred_probs, test_gt_labels,
            #                       test_pred_labels, train_result, val_result, test_result)
            #     print('save epoch {}!'.format(epoch))

        '''save final epoch results'''
        save_k_final_results(net, save_dir, save_display_dir,
                             i, lr_epoch, val_dict['val_gt_labels'], val_dict['val_pred_labels'],
                             val_dict['val_pred_probs'], cla)
        # save_k_final_results(net, save_dir, save_display_dir,
        #                      i, lr_epoch, val_dict['val_gt_labels'], val_dict['val_pred_labels'],
        #                      val_dict['val_pred_probs'], cla, test_pred_probs, test_gt_labels,
        #                      test_pred_labels)
        break


if __name__ == '__main__':
    start_time = time.time()
    print("start time:", datetime.datetime.now())

    parser = BaseConfig(
        os.path.join("./config/", "config.yaml"))
    main(parser)

    print("\nEnd time:", datetime.datetime.now())
    h, m, s = s2t(time.time() - start_time)
    print("Using Time: %02dh %02dm %02ds" % (h, m, s))
