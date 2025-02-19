import torch
import copy
from prettytable import PrettyTable


def average_weights(w, idxs_users, datanumber_client, datanumber_client_by_attr=None, islist=False):
    """
    Returns the average of the weights.
    """
    total_data_points = sum([datanumber_client[r] for r in idxs_users])
    if datanumber_client_by_attr is not None:
        datanumber_client_by_attr = torch.tensor(datanumber_client_by_attr)
        total_datanumber_client_by_attr = datanumber_client_by_attr[idxs_users].sum(0)
    
    w_avg = copy.deepcopy(w[idxs_users[0]])
    for idx in range(len(idxs_users)):
        fed_avg_freqs = datanumber_client[idxs_users[idx]] / total_data_points
        if datanumber_client_by_attr is not None:
            fed_avg_freqs_by_attr = datanumber_client_by_attr[idxs_users[idx]] / total_datanumber_client_by_attr

        if islist:
            if idx == 0:
                w_avg = w_avg * fed_avg_freqs
            else:
                w_avg += w[idxs_users[idx]] * fed_avg_freqs
        else:
            if idx == 0:
                for key in w_avg:
                    if datanumber_client_by_attr is not None and 'lora_S' in key and w_avg[key].shape[0] == len(fed_avg_freqs_by_attr):
                        w_avg[key] = w_avg[key] * fed_avg_freqs_by_attr[:, None].to(w_avg[key].device)
                    else:
                        w_avg[key] = w_avg[key] * fed_avg_freqs
            else:
                for key in w_avg:
                    if datanumber_client_by_attr is not None and 'lora_S' in key and w_avg[key].shape[0] == len(fed_avg_freqs_by_attr):
                        w_avg[key] += w[idxs_users[idx]][key] * fed_avg_freqs_by_attr[:, None].to(w[idxs_users[idx]][key].device)
                    else:
                        w_avg[key] += w[idxs_users[idx]][key] * fed_avg_freqs

    return w_avg

def average_weights_EMA(w_g, w, idxs_users, datanumber_client, datanumber_client_by_attr, epoch, max_epoch, beta=0.999, islist=False, shared_half_s=False):
    """
    Returns the Exponential Moving Average (EMA) of the weights.
    
    Parameters:
    - w_g: Current global weights (EMA) to be updated.
    - w: Local weights from each user.
    - idxs_users: List of indices of selected clients.
    - datanumber_client: Number of data points for each client.
    - datanumber_client_by_attr: Number of data points for each demographic group, num_clients x num_groups 
    - islist: Whether the weights are in list format.
    - beta: Decay rate for EMA, with typical values close to 1 (e.g., 0.999).
    
    Returns:
    - Updated EMA weights.
    """
    total_data_points = sum([datanumber_client[r] for r in idxs_users])
    datanumber_client_by_attr = torch.tensor(datanumber_client_by_attr)
    total_datanumber_client_by_attr = datanumber_client_by_attr[idxs_users].sum(0)
    
    w_avg = copy.deepcopy(w[idxs_users[0]])
    for idx in range(len(idxs_users)):
        fed_avg_freqs = datanumber_client[idxs_users[idx]] / total_data_points
        fed_avg_freqs_by_attr = datanumber_client_by_attr[idxs_users[idx]] / total_datanumber_client_by_attr
        
        if islist:
            if idx == 0:
                w_avg = w_avg * fed_avg_freqs
            else:
                w_avg += w[idxs_users[idx]] * fed_avg_freqs
        else:
            if idx == 0:
                for key in w_avg:
                    if 'lora_S' in key and w_avg[key].shape[0] == len(fed_avg_freqs_by_attr):
                        w_avg[key] = w_avg[key] * fed_avg_freqs_by_attr[:, None].to(w_avg[key].device)
                    else:
                        w_avg[key] = w_avg[key] * fed_avg_freqs
            else:
                for key in w_avg:
                    if 'lora_S' in key and w_avg[key].shape[0] == len(fed_avg_freqs_by_attr):
                        w_avg[key] += w[idxs_users[idx]][key] * fed_avg_freqs_by_attr[:, None].to(w[idxs_users[idx]][key].device)
                    else:
                        w_avg[key] += w[idxs_users[idx]][key] * fed_avg_freqs
    
    beta_decay = beta * (epoch / max(max_epoch, 1))
    for key in w_avg:
        if shared_half_s and 'lora_S' in key and w_avg[key].shape[0] == len(fed_avg_freqs_by_attr):
            n_groups, n_dim = w_avg[key].shape
            S_aft = torch.cat(
                [torch.mean(w_avg[key][:, :n_dim//2], dim=0, keepdim=True).repeat(n_groups,1),
                w_avg[key][:, n_dim//2:]], dim=1
            )
            w_avg[key] = S_aft

        w_avg[key] = (1 - beta_decay) * w_avg[key] + beta_decay * w_g[key]

    return w_avg


def count_parameters(model, model_name):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if model_name in name:
            # if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params