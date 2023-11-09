def update_trainers(learning_networks, target_networks, tau):
    """
    update the trainers_tar par using the trainers_cur
    This way is not the same as copy_, but the result is the same
    out:
    |agents_tar: the agents with new par updated towards agents_current
    """
    for learning_net, target_net in zip(learning_networks, target_networks):
        key_list = list(learning_net.state_dict().keys())
        state_dict_t = target_net.state_dict()
        state_dict_c = learning_net.state_dict()
        for key in key_list:
            state_dict_t[key] = state_dict_c[key]*tau + \
                    (1-tau)*state_dict_t[key] 
        target_net.load_state_dict(state_dict_t)
    return target_networks