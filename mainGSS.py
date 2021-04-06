from backbone import ResNet12, ConvNet
from GSS import GSS
from GSS_utils import set_logging_config, adjust_learning_rate, save_checkpoint, allocate_tensors, preprocessing, \
    initialize_nodes_edges, backbone_two_stage_initialization, one_hot_encode
from dataloader import Cifar, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
import logging
import argparse
import imp
import nni


class GSSTrainer(object):
    def __init__(self, enc_module, gnn_module, data_loader, log, arg, config, best_step, nni_flag=False):
        """
        The Trainer of DPGN model
        :param enc_module: backbone network (Conv4, ResNet12, ResNet18, WRN)
        :param gnn_module: DPGN model
        :param data_loader: data loader
        :param log: logger
        :param arg: command line arguments
        :param config: model configurations
        :param best_step: starting step (step at best eval acc or 0 if starts from scratch)
        """
        import nni
        self.arg = arg
        self.config = config
        self.train_opt = config['train_config']
        self.eval_opt = config['eval_config']
        self.nni_flag = nni_flag

        # initialize variables
        self.tensors = allocate_tensors()
        for key, tensor in self.tensors.items():
            self.tensors[key] = tensor.to(self.arg.device)

        # set backbone and DPGN
        self.enc_module = enc_module.to(arg.device)
        self.gnn_module = gnn_module.to(arg.device)

        # set logger
        self.log = log

        # get data loader
        self.data_loader = data_loader

        # set parameters
        self.module_params = list(self.enc_module.parameters()) + list(self.gnn_module.parameters())

        # set optimizer
        self.optimizer = optim.Adam(
            params=self.module_params,
            lr=self.train_opt['lr'],
            weight_decay=self.train_opt['weight_decay'])

        # set loss
        self.edge_loss = nn.BCELoss(reduction='none')
        self.pred_loss = nn.CrossEntropyLoss(reduction='none')

        # initialize other global variables
        self.global_step = best_step
        self.best_step = best_step
        self.val_acc = 0
        self.test_acc = 0

    def train(self):
        """
        train function
        :return: None
        """

        num_supports, num_samples, query_edge_mask, evaluation_mask = \
            preprocessing(self.train_opt['num_ways'],
                          self.train_opt['num_shots'],
                          self.train_opt['num_queries'],
                          self.train_opt['batch_size'],
                          self.arg.device)

        # main training loop, batch size is the number of tasks
        for iteration, batch in enumerate(self.data_loader['train']()):
            # init grad
            self.optimizer.zero_grad()

            # set current step
            self.global_step += 1

            # initialize nodes and edges for dual graph model
            support_data, support_label, query_data, query_label, all_data,\
            edge_GT, node_feature_L, edge_feature_H, edge_feature_L = initialize_nodes_edges(batch,
                                                                      num_supports,
                                                                      self.tensors,
                                                                      self.train_opt['batch_size'],
                                                                      self.train_opt['num_queries'],
                                                                      self.train_opt['num_ways'],
                                                                      self.arg.device)

            # print(torch.where(edge_feature_L != 0.0), edge_feature_L.shape)

            # set as train mode
            self.enc_module.train()
            self.gnn_module.train()
            # print(all_data.shape)
            # use backbone encode image
            last_layer_data = backbone_two_stage_initialization(all_data, self.enc_module)

            # run the GSS model
            # print(last_layer_data.shape)
            # exit()
            VSH, VSH_metric, VSL, VSL_metric = self.gnn_module(last_layer_data,
                                                               node_feature_L,
                                                               edge_feature_H,
                                                               edge_feature_L)

            # for vl in VSL:
            #     print(torch.where(vl !=0.0), vl.shape)
            # exit()

            # compute loss
            total_loss, query_node_cls_acc_generations, query_edge_loss_generations = \
                self.compute_train_loss_pred(edge_GT,
                                             VSH,
                                             VSH_metric,
                                             VSL,
                                             VSL_metric,
                                             query_edge_mask,
                                             evaluation_mask,
                                             num_supports,
                                             support_label,
                                             query_label)

            # back propagation & update
            total_loss.backward()
            self.optimizer.step()

            # adjust learning rate
            adjust_learning_rate(optimizers=[self.optimizer],
                                 lr=self.train_opt['lr'],
                                 iteration=self.global_step,
                                 dec_lr_step=self.train_opt['dec_lr'],
                                 lr_adj_base =self.train_opt['lr_adj_base'])


            if self.global_step % self.arg.log_step == 0:
                self.log.info('step : {}  train_edge_loss : {}  node_acc : {}'.format(
                    self.global_step,
                    query_edge_loss_generations[-1],
                    query_node_cls_acc_generations[-1]))

            # evaluation
            if self.global_step % self.eval_opt['interval'] == 0:
                is_best = 0
                test_acc = self.eval(partition='test')
                if test_acc > self.test_acc:
                    is_best = 1
                    self.test_acc = test_acc
                    self.best_step = self.global_step

                # log evaluation info
                self.log.info('test_acc : {}         step : {} '.format(test_acc, self.global_step))
                self.log.info('test_best_acc : {}    step : {}'.format( self.test_acc, self.best_step))
                if self.nni_flag:
                    nni.report_intermediate_result(test_acc)

                # save checkpoints (best and newest)
                save_checkpoint({
                    'iteration': self.global_step,
                    'enc_module_state_dict': self.enc_module.state_dict(),
                    'gnn_module_state_dict': self.gnn_module.state_dict(),
                    'test_acc': self.test_acc,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best, os.path.join(self.arg.checkpoint_dir, self.arg.exp_name))

        if self.nni_flag:
            nni.report_final_result(self.test_acc)

    def eval(self, partition='test', log_flag=True):
        """
        evaluation function
        :param partition: which part of data is used
        :param log_flag: if log the evaluation info
        :return: None
        """

        num_supports, num_samples, query_edge_mask, evaluation_mask = preprocessing(
            self.eval_opt['num_ways'],
            self.eval_opt['num_shots'],
            self.eval_opt['num_queries'],
            self.eval_opt['batch_size'],
            self.arg.device)

        query_edge_loss_generations = []
        query_node_cls_acc_generations = []
        # main training loop, batch size is the number of tasks
        for current_iteration, batch in enumerate(self.data_loader[partition]()):

            # initialize nodes and edges for dual graph model
            support_data, support_label, query_data, query_label, all_data, \
            edge_GT, node_feature_L, edge_feature_H, edge_feature_L = initialize_nodes_edges(batch,
                                                                                             num_supports,
                                                                                             self.tensors,
                                                                                             self.eval_opt['batch_size'],
                                                                                             self.eval_opt['num_queries'],
                                                                                             self.eval_opt['num_ways'],
                                                                                             self.arg.device)

            # set as eval mode
            self.enc_module.eval()
            self.gnn_module.eval()

            last_layer_data = backbone_two_stage_initialization(all_data, self.enc_module)

            # run the DPGN model
            VSH, VSH_metric, VSL, VSL_metric = self.gnn_module(last_layer_data,
                                                               node_feature_L,
                                                               edge_feature_H,
                                                               edge_feature_L)

            edge_loss, node_acc = self.compute_eval_loss_pred(edge_GT,
                                                              VSL,
                                                              query_edge_mask,
                                                              evaluation_mask,
                                                              num_supports,
                                                              support_label,
                                                              query_label)


            query_edge_loss_generations.append(edge_loss)
            query_node_cls_acc_generations.append(node_acc)

        # logging
        if log_flag:
            # query_edge_loss_generations = [i.item() for i in query_edge_loss_generations]
            # print(query_edge_loss_generations)
            # print(np.array(query_edge_loss_generations))
            self.log.info('------------------------------------')
            self.log.info('step : {}  {}_edge_loss : {}  {}_node_acc : {}'.format(
                self.global_step, partition,
                np.array(query_edge_loss_generations).mean(),
                partition,
                np.array(query_node_cls_acc_generations).mean()))

            self.log.info('evaluation: total_count=%d, accuracy: mean=%.2f%%, std=%.2f%%, ci95=%.2f%%' %
                          (current_iteration,
                           np.array(query_node_cls_acc_generations).mean() * 100,
                           np.array(query_node_cls_acc_generations).std() * 100,
                           1.96 * np.array(query_node_cls_acc_generations).std()
                           / np.sqrt(float(len(np.array(query_node_cls_acc_generations)))) * 100))
            self.log.info('------------------------------------')

        return np.array(query_node_cls_acc_generations).mean()

    def compute_train_loss_pred(self,
                                edge_GT,
                                VSHs,
                                VSHs_metric,
                                VSLs,
                                VSLs_metric,
                                query_edge_mask,
                                evaluation_mask,
                                num_supports,
                                support_label,
                                query_label):
        """

        :param edge_GT:
        :param VSH:
        :param VSH_metric:
        :param VSL:
        :param VSL_metric:
        :param query_edge_mask:
        :param evaluation_mask:
        :param num_supports:
        :param support_label:
        :param query_label:
        :return:
        """

        #High Space Edge Loss
        edge_losses_high = [self.balanced_edge_loss(VSH, edge_GT, query_edge_mask, evaluation_mask) for VSH in VSHs]
        # Low Space Edge loss
        edge_losses_low = [self.balanced_edge_loss(VSL, edge_GT, query_edge_mask, evaluation_mask) for VSL in VSLs]

        edge_losses = [0.0*edge_loss_high + 1.0 * edge_loss_low for edge_loss_high, edge_loss_low in zip(edge_losses_high, edge_losses_low)]

        # High Space Node Classification Loss
        query_node_las_high = [self.node_loss_acc(VSH, query_label, num_supports, support_label)
                               for VSH in VSHs]
        query_node_las_high_metric = [self.node_loss_acc(VSH_metric, query_label, num_supports, support_label)
                                      for VSH_metric in VSHs_metric]
        query_node_losses_high = [la[0] for la in query_node_las_high_metric]
        query_node_accs_high = [la[1] for la in query_node_las_high]

        # Low Space Node Classification Loss
        query_node_las_low = [self.node_loss_acc(VSL, query_label, num_supports, support_label)
                               for VSL in VSLs]
        query_node_las_low_metric = [self.node_loss_acc(VSL_metric, query_label, num_supports, support_label)
                                      for VSL_metric in VSLs_metric]
        query_node_losses_low = [la[0] for la in query_node_las_low_metric]
        query_node_accs_low = [la[1] for la in query_node_las_low]

        # total loss
        total_losses = [edge_loss + 0.1 * node_loss for edge_loss, node_loss in zip(edge_losses, query_node_losses_low)]

        # compute total loss
        total_loss = []
        num_loss = self.config['num_loss_generation']
        for l in range(num_loss - 1):
            total_loss += [total_losses[l].view(-1) * self.config['generation_weight']]
        total_loss += [total_losses[-1].view(-1) * 1.0]
        total_loss = torch.mean(torch.cat(total_loss, 0))
        return total_loss, query_node_accs_low, edge_losses_low

    def compute_eval_loss_pred(self,
                               edge_GT,
                               edge_pred,
                               query_edge_mask,
                               evaluation_mask,
                               num_supports,
                               support_label,
                               query_label):
        """
        compute the query classification loss and query classification accuracy
        :param query_edge_losses: container for losses of queries' edges
        :param query_node_accs: container for classification accuracy of queries
        :param all_label_in_edge: ground truth label in edge form of point graph
        :param point_similarities: prediction edges of point graph
        :param query_edge_mask: mask for queries
        :param evaluation_mask: mask for evaluation (for unsupervised setting)
        :param num_supports: number of samples in support set
        :param support_label: label of support set
        :param query_label: label of query set
        :return: query classification loss
                 query classification accuracy
        """

        point_similarity = edge_pred[-1]
        query_edge_loss = self.balanced_edge_loss(point_similarity, edge_GT,query_edge_mask, evaluation_mask)

        # prediction
        _, query_node_acc = self.node_loss_acc(point_similarity, query_label, num_supports, support_label)

        return query_edge_loss.item(), query_node_acc.item()

    def balanced_edge_loss(self, pred, target, query_edge_mask, evaluation_mask):
        loss = self.edge_loss(1.0-pred, 1.0-target)
        posi_loss = torch.sum(loss * query_edge_mask * target * evaluation_mask)\
                    / torch.sum(query_edge_mask * target * evaluation_mask)
        nega_loss = torch.sum(loss * query_edge_mask * (1 - target) * evaluation_mask)\
                    / torch.sum(query_edge_mask * (1 - target) * evaluation_mask)
        return posi_loss + nega_loss

    def node_loss_acc(self, edge, query_label, num_supports, support_label):
        pred = torch.bmm(edge[:, num_supports:, :],
                         one_hot_encode(self.train_opt['num_ways'], support_label.long(), self.arg.device))
        loss = self.pred_loss(pred, query_label.long()).mean()
        acc = torch.eq(torch.max(pred, -1)[1], query_label.long()).float().mean()

        # pred = pred.flatten(end_dim=-2)
        # target = query_label.flatten()
        # loss = self.pred_loss(pred, target.long()).mean()
        # acc = torch.eq(torch.max(pred, -1)[1], target.long()).float().mean()

        return [loss, acc]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default=None,
                        help='note this experiment as your wish')

    parser.add_argument('--device', type=str, default='cuda:0',
                        help='gpu device number of using')

    parser.add_argument('--config', type=str, default='config.py',
                        help='config file with parameters of the experiment. '
                             'It is assumed that the config file is placed under the directory ./config')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='path that checkpoint will be saved and loaded. '
                             'It is assumed that the checkpoint file is placed under the directory ./checkpoints')

    parser.add_argument('--num_gpu', type=int, default=1,
                        help='number of gpu')

    parser.add_argument('--display_step', type=int, default=100,
                        help='display training information in how many step')

    parser.add_argument('--log_step', type=int, default=100,
                        help='log information in how many steps')

    parser.add_argument('--log_dir', type=str, default='logs',
                        help='path that log will be saved. '
                             'It is assumed that the checkpoint file is placed under the directory ./logs')

    parser.add_argument('--dataset_root', type=str, default='./DATA',
                        help='root directory of dataset')

    parser.add_argument('--seed', type=int, default=222,
                        help='random seed')

    parser.add_argument('--mode', type=str, default='train',
                        help='train or eval')

    args_opt = parser.parse_args()

    config_file = args_opt.config

    # Set train and test datasets and the corresponding data loaders
    config = imp.load_source("", config_file).config
    train_opt = config['train_config']
    eval_opt = config['eval_config']

    args_opt.exp_name = '_'+args_opt.exp_name if args_opt.exp_name is not None else args_opt.exp_name
    args_opt.exp_name = '{}way_{}shot_{}_{}{}'.format(train_opt['num_ways'],
                                                      train_opt['num_shots'],
                                                      config['backbone'],
                                                      config['dataset_name'],
                                                      args_opt.exp_name)
    train_opt['num_queries'] = 1
    eval_opt['num_queries'] = 1
    set_logging_config(os.path.join(args_opt.log_dir, args_opt.exp_name))
    logger = logging.getLogger('main')
    # set_logging_config(os.path.join(args_opt.log_dir, args_opt.exp_name), logger)
    # logger.setLevel(logging.INFO)
    # print(logger)
    # exit()

    # Load the configuration params of the experiment
    logger.info('Launching experiment from: {}'.format(config_file))
    logger.info('Generated logs will be saved to: {}'.format(args_opt.log_dir))
    logger.info('Generated checkpoints will be saved to: {}'.format(args_opt.checkpoint_dir))
    print()

    logger.info('-------------command line arguments-------------')
    logger.info(args_opt)
    print()
    logger.info('-------------configs-------------')
    logger.info(config)

    # set random seed
    np.random.seed(args_opt.seed)
    torch.manual_seed(args_opt.seed)
    torch.cuda.manual_seed_all(args_opt.seed)
    random.seed(args_opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if config['dataset_name'] == 'mini-imagenet':
        dataset = MiniImagenet
        print('Dataset: MiniImagenet')
    elif config['dataset_name'] == 'tiered-imagenet':
        dataset = TieredImagenet
        print('Dataset: TieredImagenet')
    elif config['dataset_name'] == 'cifar-fs':
        dataset = Cifar
        print('Dataset: Cifar')
    elif config['dataset_name'] == 'cub-200-2011':
        dataset = CUB200
        print('Dataset: CUB200')
    else:
        logger.info('Invalid dataset: {}, please specify a dataset from '
                    'mini-imagenet, tiered-imagenet, cifar-fs and cub-200-2011.'.format(config['dataset_name']))
        exit()

    cifar_flag = True if args_opt.exp_name.__contains__('cifar') else False
    if config['backbone'] == 'resnet12':
        enc_module = ResNet12(emb_size=config['emb_size'], cifar_flag=cifar_flag)
        print('Backbone: ResNet12')
    elif config['backbone'] == 'convnet':
        enc_module = ConvNet(emb_size=config['emb_size'], cifar_flag=cifar_flag)
        print('Backbone: ConvNet')
    else:
        logger.info('Invalid backbone: {}, please specify a backbone model from '
                    'convnet or resnet12.'.format(config['backbone']))
        exit()

    gnn_module = GSS(config['num_generation'],
                      train_opt['dropout'],
                      config['emb_size'],
                      train_opt['num_ways'] * train_opt['num_shots'],
                      train_opt['num_ways'] * train_opt['num_shots'] + train_opt['num_ways'] * train_opt['num_queries'],
                      train_opt['loss_indicator'],
                      config['point_distance_metric'],
                      config['distribution_distance_metric'],
                      train_opt['high_scale'],
                      train_opt['low_scale'])

    # multi-gpu configuration
    [print('GPU: {}  Spec: {}'.format(i, torch.cuda.get_device_name(i))) for i in range(args_opt.num_gpu)]

    if args_opt.num_gpu > 1:
        print('Construct multi-gpu model ...')
        enc_module = nn.DataParallel(enc_module, device_ids=range(args_opt.num_gpu), dim=0)
        gnn_module = nn.DataParallel(gnn_module, device_ids=range(args_opt.num_gpu), dim=0)
        print('done!\n')

    if not os.path.exists(os.path.join(args_opt.checkpoint_dir, args_opt.exp_name)):
        os.makedirs(os.path.join(args_opt.checkpoint_dir, args_opt.exp_name))
        logger.info('no checkpoint for model: {}, make a new one at {}'.format(
            args_opt.exp_name,
            os.path.join(args_opt.checkpoint_dir, args_opt.exp_name)))
        best_step = 0
    else:
        if not os.path.exists(os.path.join(args_opt.checkpoint_dir, args_opt.exp_name, 'model_best.pth.tar')):
            best_step = 0
        else:
            logger.info('find a checkpoint, loading checkpoint from {}'.format(
                os.path.join(args_opt.checkpoint_dir, args_opt.exp_name)))
            best_checkpoint = torch.load(os.path.join(args_opt.checkpoint_dir, args_opt.exp_name, 'model_best.pth.tar'))

            logger.info('best model pack loaded')
            best_step = best_checkpoint['iteration']
            enc_module.load_state_dict(best_checkpoint['enc_module_state_dict'])
            gnn_module.load_state_dict(best_checkpoint['gnn_module_state_dict'])
            logger.info('current best test accuracy is: {}, at step: {}'.format(best_checkpoint['test_acc'], best_step))

    dataset_train = dataset(root=args_opt.dataset_root, partition='train')
    dataset_valid = dataset(root=args_opt.dataset_root, partition='val')
    dataset_test = dataset(root=args_opt.dataset_root, partition='test')

    train_loader = DataLoader(dataset_train,
                              num_tasks=train_opt['batch_size'],
                              num_ways=train_opt['num_ways'],
                              num_shots=train_opt['num_shots'],
                              num_queries=train_opt['num_queries'],
                              epoch_size=train_opt['iteration'])
    valid_loader = DataLoader(dataset_valid,
                              num_tasks=eval_opt['batch_size'],
                              num_ways=eval_opt['num_ways'],
                              num_shots=eval_opt['num_shots'],
                              num_queries=eval_opt['num_queries'],
                              epoch_size=eval_opt['iteration'])
    test_loader = DataLoader(dataset_test,
                             num_tasks=eval_opt['batch_size'],
                             num_ways=eval_opt['num_ways'],
                             num_shots=eval_opt['num_shots'],
                             num_queries=eval_opt['num_queries'],
                             epoch_size=eval_opt['iteration'])

    data_loader = {'train': train_loader,
                   'val': valid_loader,
                   'test': test_loader}

    # create trainer
    trainer = GSSTrainer(enc_module=enc_module,
                          gnn_module=gnn_module,
                          data_loader=data_loader,
                          log=logger,
                          arg=args_opt,
                          config=config,
                          best_step=best_step)

    if args_opt.mode == 'train':
        trainer.train()
    elif args_opt.mode == 'eval':
        trainer.eval()
    else:
        print('select a mode')
        exit()


if __name__ == '__main__':
    main()
