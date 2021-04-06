import torch.nn as nn
import torch.nn.functional as F
import torch


def get_diagonal_mask(edge_feature):
    """

    :param edge_feature: T,N_s+N_q,N_s
    :return:
    """
    batch, i, j = edge_feature.size()
    device = edge_feature.device

    mask1 = torch.eye(j).to(device)
    mask2 = torch.zeros(i-j,j).to(device)
    mask = torch.cat([mask1, mask2], dim=0)
    batch_mask = mask.unsqueeze(0).repeat(batch,1,1)
    return batch_mask


class VertexSimilarity(nn.Module):
    def __init__(self, Cin, Cbase, dropout=0.0, scale=1.0):
        super(VertexSimilarity, self).__init__()
        self.Cin = Cin
        self.Cbase = Cbase
        self.dropout = dropout
        self.scale=scale
        layer_list = []

        layer_list += [nn.Conv2d(in_channels=self.Cin, out_channels=self.Cbase * 2, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.Cbase * 2),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        layer_list += [nn.Conv2d(in_channels=self.Cbase * 2, out_channels=self.Cbase, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.Cbase),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        layer_list += [nn.Conv2d(in_channels=self.Cbase, out_channels=1, kernel_size=1)]
        self.point_sim_transform = nn.Sequential(*layer_list)

    def forward(self, edge_feature, node_feature, metric='l1'):
        """
        calculate node similarity based on node feature and previous edge feature
        :param node_fature: T,N,C
        :param edge_feature: T,N,N
        :param metric:
        :return: edge feature
        """
        num_supports = edge_feature.size(2)
        V_s = node_feature[:,:num_supports,:].contiguous().unsqueeze(1)
        V_all = node_feature.unsqueeze(2)
        if metric == 'l2':
            V_similarity = (V_all - V_s)**2
        elif metric == 'l1':
            V_similarity = torch.abs(V_all - V_s)

        V_simi_trans = V_similarity.permute(0,3,1,2) # T,N_s+N_q,N_s,C --> T,C,N_s+N_q,N_s

        edge_new = torch.sigmoid(self.point_sim_transform(V_simi_trans)).squeeze(1)

        # normalization
        diagonal_mask= get_diagonal_mask(edge_feature)
        edge_last = edge_feature * (1.0 - diagonal_mask)
        edge_last_sum = edge_last.sum(dim=2, keepdim=True)
        edge_new = F.normalize(edge_new*edge_last, p=1, dim=2) * edge_last_sum
        edge_new += (diagonal_mask + 1e-6)
        edge_new /= edge_new.sum(dim=2, keepdim=True)

        # similarity_metric = -torch.mean(V_similarity, 3)*self.scale
        similarity_metric = -torch.sum(V_similarity, 3) * self.scale
        # print(V_similarity.shape)
        # print(self.scale)

        return edge_new, similarity_metric


class VertexUpdateLow(nn.Module):
    def __init__(self, Cin, Cout):
        super(VertexUpdateLow, self).__init__()
        self.Cout = Cout
        self.transform = nn.Sequential(*[nn.Linear(in_features=Cin, out_features=Cout, bias=True),
                                         nn.LeakyReLU()])

    def forward(self, edge_feature_high, node_feature_low):
        T, N_sq, N_s = edge_feature_high.size()
        inFeature = torch.cat([edge_feature_high, node_feature_low], dim=2).view(T*N_sq,N_s*2)
        outFeature = self.transform(inFeature)
        new_node_feature_low = outFeature.view(T, N_sq, N_s)

        return new_node_feature_low


class VertexUpdateHigh(nn.Module):
    def __init__(self, Cin, Cout, dropout=0.0):
        super(VertexUpdateHigh, self).__init__()
        self.Cin = Cin
        self.Cout = Cout
        self.dropout = dropout
        layer_list = []
        layer_list += [nn.Conv2d(in_channels=self.Cin, out_channels=self.Cout * 2, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.Cout * 2),
                       nn.LeakyReLU()]

        layer_list += [nn.Conv2d(in_channels=self.Cout * 2, out_channels=self.Cout, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.Cout),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        self.transform = nn.Sequential(*layer_list)

    def forward(self, edge_feature_low, node_feature_high):
        T, N_sq, N_s = edge_feature_low.size()
        diagonal_mask = get_diagonal_mask(edge_feature_low)
        edge_feat = F.normalize(edge_feature_low * (1.0 - diagonal_mask), p=1, dim=-1)
        aggr_feat_all = torch.bmm(edge_feat, node_feature_high[:,:N_s,:]) # T, N_sq, C
        aggr_feat_support = torch.bmm(edge_feat.transpose(1,2), node_feature_high) # T, N-s, C
        aggr_feat_all[:,:N_s,:] = aggr_feat_support

        inFeature = torch.cat([node_feature_high, aggr_feat_all], -1).transpose(1,2)
        outFeature = self.transform(inFeature.unsqueeze(-1))
        new_node_feature_high = outFeature.transpose(1,2).squeeze(-1)

        return new_node_feature_high


class GSS(nn.Module):
    def __init__(self, num_generations, dropout, emb_size, num_support_sample, num_sample, loss_indicator, high_metric, low_metric, high_scale, low_scale):
        super(GSS, self).__init__()
        self.num_gen = num_generations
        self.dropout = dropout
        self.num_support = num_support_sample
        self.num_sample = num_sample
        self.loss_indicator = loss_indicator
        self.high_metric = high_metric
        self.low_metric = low_metric
        self.high_scale = high_scale
        self.los_scale = low_scale

        V_Sim_H = VertexSimilarity(emb_size, emb_size, dropout=self.dropout)
        self.add_module('initial_edge_high', V_Sim_H)
        for l in range(self.num_gen):
            VSH = VertexSimilarity(emb_size, emb_size, dropout=self.dropout if l < self.num_gen-1 else 0.0, scale=high_scale)
            VUL = VertexUpdateLow(2*num_support_sample, num_support_sample)
            VSL = VertexSimilarity(num_support_sample, num_support_sample, dropout=self.dropout if l < self.num_gen-1 else 0.0, scale=low_scale)
            VUH = VertexUpdateHigh(2*emb_size, emb_size)

            self.add_module('Vertex_Simi_High_{}'.format(l), VSH)
            self.add_module('Vertex_Update_Low_{}'.format(l), VUL)
            self.add_module('Vertex_Simi_Low_{}'.format(l), VSL)
            self.add_module('Vertex_Update_High_{}'.format(l), VUH)

    def forward(self, node_high, node_low, edge_high, edge_low):
        simi_high = []
        simi_high_metric = []
        simi_low = []
        simi_low_metric = []

        edge_high, _ = self._modules['initial_edge_high'](edge_high, node_high, self.high_metric)
        for l in range(self.num_gen):
            edge_high, simi_metric_high = self._modules['Vertex_Simi_High_{}'.format(l)](edge_high, node_high, self.high_metric)
            node_low = self._modules['Vertex_Update_Low_{}'.format(l)](edge_high, node_low)
            edge_low, simi_metric_low = self._modules['Vertex_Simi_Low_{}'.format(l)](edge_low, node_low, self.low_metric)
            node_high = self._modules['Vertex_Update_High_{}'.format(l)](edge_low, node_high)

            simi_high.append(edge_high * self.loss_indicator[0])
            simi_high_metric.append(simi_metric_high * self.loss_indicator[1])
            simi_low.append(edge_low * self.loss_indicator[2])
            simi_low_metric.append(simi_metric_low * self.loss_indicator[3])
        # exit()
        return simi_high, simi_high_metric, simi_low, simi_low_metric



if __name__ == '__main__':
    num_generations, dropout, emb_size, num_support_sample, num_sample, loss_indicator, high_metric, low_metric = \
    2, 0.1, 8, 5, 6, [1,1,1,1], 'l1', 'l1'

    net = GSS(num_generations, dropout, emb_size, num_support_sample, num_sample, loss_indicator, high_metric, low_metric)
    # print(net)
    net.cuda()

    node_high, node_low, edge_high, edge_low = \
    torch.randn(4,6,8), torch.randn(4,6,5), torch.randn(4,6,5), torch.randn(4,6,5)

    for i in [node_high, node_low, edge_high, edge_low]:
        i.requires_grad = True

    node_high, node_low, edge_high, edge_low = [i.cuda() for i in [node_high, node_low, edge_high, edge_low]]

    out = net(node_high, node_low, edge_high, edge_low)
    for o in out:
        print(len(o))
        print(o[0].shape)
        o[0].sum().backward(retain_graph=True)

    print('OK')
