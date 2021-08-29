import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
#warnings.filterwarnings('ignore')

# STAN model modified for SIRVC model (SIR + vaccinations + community interactions) and edge weights

class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        self.E = None
        self.fc = nn.Linear(in_dim, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        if self.E != None:
            e = F.leaky_relu(a) * torch.unsqueeze(self.E, 1)
            return {'e': e}
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}
 
    def forward(self, h, E):
        self.E = E # adjacency matrix; edge weights
        z = self.fc(h.float())
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')

class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h, E):
        head_outs = [attn_head(h, E) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs))
        
class STAN(nn.Module):
    def __init__(self, g, in_dim, hidden_dim1, hidden_dim2, gru_dim, num_heads, pred_window, device):
        super(STAN, self).__init__()
        self.g = g
        
        self.layer1 = MultiHeadGATLayer(self.g, in_dim, hidden_dim1, num_heads)
        self.layer2 = MultiHeadGATLayer(self.g, hidden_dim1 * num_heads, hidden_dim2, 1)

        self.pred_window = pred_window
        self.gru = nn.GRUCell(hidden_dim2, gru_dim)
    
        self.nn_res_I = nn.Linear(gru_dim+2, pred_window)
        self.nn_res_R = nn.Linear(gru_dim+2, pred_window)

        self.nn_res_sir = nn.Linear(gru_dim+2, 3) # predict three variables
        
        self.hidden_dim2 = hidden_dim2
        self.gru_dim = gru_dim
        self.device = device

    def forward(self, dynamic, cI, cR, N, I, R, h=None, V=None, e_weights=None):
        '''
        dynamic: an array of the dI/dR/dS values for each loc for every day from the start date to the end date
                of the input data (e.g., training dates) jumping by slide_step each time.
                ex:[[[dI, dR, dS, * history_window size], * num dates] * num_locs]
        cI and cR: hold the dI or dR values at each of the end days for a history_window jumping by slide step
                each time, for a single location
                ex: [dI, * num dates]
        N: population size of location
        I and R: hold the total I or R values at each of the end days (format matches cI and cR)
                ex: [I, * num dates]
        '''
        num_loc, timestep, n_feat = dynamic.size()
        N = N.squeeze()

        if h is None:
            h = torch.zeros(1, self.gru_dim).to(self.device)
            gain = nn.init.calculate_gain('relu')
            nn.init.xavier_normal_(h, gain=gain)  

        new_I = []
        new_R = []
        phy_I = []
        phy_R = []
        self.alpha_list = []
        self.beta_list = []
        #self.theta_list = []
        self.alpha_scaled = []
        self.beta_scaled = []
        #self.theta_scaled = [] 
        '''
        for every time period, e.g., days 0-5, 5-10, 10-15..., send the corresponding date
        range for all locations with all features into layer 1, then pass through activation
        function, then repeat with layer 2
        '''
        for each_step in range(timestep):
            if e_weights is not None: 
                cur_h = self.layer1(dynamic[:, each_step, :], e_weights[each_step])
            else:
                cur_h = self.layer1(dynamic[:, each_step, :], None)
            cur_h = F.elu(cur_h)
            cur_h = self.layer2(cur_h, None)
            cur_h = F.elu(cur_h)
            
            cur_h = torch.max(cur_h, 0)[0].reshape(1, self.hidden_dim2)
            
            h = self.gru(cur_h, h)
            hc = torch.cat((h, cI[each_step].reshape(1,1), cR[each_step].reshape(1,1)),dim=1)
            
            pred_I = self.nn_res_I(hc)
            pred_R = self.nn_res_R(hc)
            new_I.append(pred_I)
            new_R.append(pred_R)

            pred_res = self.nn_res_sir(hc)
            alpha = pred_res[:, 0] # beta of SIR model
            beta =  pred_res[:, 1] # gamma of SIR model
            #theta = pred_res[:, 2] # theta of SIR model
            
            self.alpha_list.append(alpha)
            self.beta_list.append(beta)
            #self.theta_list.append(theta)
            alpha = torch.sigmoid(alpha)
            beta = torch.sigmoid(beta)
            #theta = torch.sigmoid(theta)
            self.alpha_scaled.append(alpha)
            self.beta_scaled.append(beta)
            #self.theta_scaled.append(theta)

            cur_phy_I = []
            cur_phy_R = []
            for i in range(self.pred_window):
                last_I = I[each_step] if i == 0 else last_I + dI.detach()
                last_R = R[each_step] if i == 0 else last_R + dR.detach()

                last_S = N - last_I - last_R
                
                dI = alpha * last_I * (last_S/N) - beta * last_I #+ theta * alpha * last_S
                dR = beta * last_I
                if V is not None:
                    dR += V[each_step][i]*0.94
                    # V needs to be in the form [[history_window * pred_window], [next time step]...]
                cur_phy_I.append(dI)
                cur_phy_R.append(dR)
            cur_phy_I = torch.stack(cur_phy_I).to(self.device).permute(1,0)
            cur_phy_R = torch.stack(cur_phy_R).to(self.device).permute(1,0)

            phy_I.append(cur_phy_I)
            phy_R.append(cur_phy_R)

        new_I = torch.stack(new_I).to(self.device).permute(1,0,2)
        new_R = torch.stack(new_R).to(self.device).permute(1,0,2)
        phy_I = torch.stack(phy_I).to(self.device).permute(1,0,2)
        phy_R = torch.stack(phy_R).to(self.device).permute(1,0,2)

        self.alpha_list = torch.stack(self.alpha_list).squeeze()
        self.beta_list = torch.stack(self.beta_list).squeeze()
        #self.theta_list = torch.stack(self.theta_list).squeeze()
        self.alpha_scaled = torch.stack(self.alpha_scaled).squeeze()
        self.beta_scaled = torch.stack(self.beta_scaled).squeeze()
        #aelf.theta_scaled = torch.stack(self.theta_scaled).squeeze()
        return new_I, new_R, phy_I, phy_R, h