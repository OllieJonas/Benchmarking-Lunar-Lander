import torch.optim as optim


def hard_copy(from_net, to_net):
    to_net.load_state_dict(dict(from_net.named_parameters()))


def soft_copy(from_net, to_net, tau):
    state_dict = dict(from_net.named_parameters())
    target_state_dict = dict(to_net.named_parameters())

    for _n in state_dict:
        state_dict[_n] = tau * state_dict[_n].clone() + (1 - tau) * target_state_dict[_n].clone()

    to_net.load_state_dict(state_dict)


def with_optim(net, lr, device=None):
    return net if device is None else net.to(device), optim.Adam(params=net.parameters(), lr=lr)
