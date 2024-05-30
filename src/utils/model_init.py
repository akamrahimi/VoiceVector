import torch 

# initialize pytorch model from checkpoint
def init_from_checkpoint(model, init_from, paths):
    if 'demucs' in init_from:
        model = load_demucs_weights(model, paths.demucs)
        return model
    
    if 'tacotron2' in init_from:
        model = load_tacotron2_weights(model, paths.tacotron2)
        return model
    
    if '.pth' in init_from:
        model = load_pth_model_weights(model, init_from)
        return model
    
    if 'avem' in init_from:
        model = load_model_weights(model, paths.voicformerAVnew)
        model = load_demucs_weights(model, paths.speaker_embedding,'net.speaker_embedding.')
        
        return model
    
    if 'voiceformerav' == init_from:
        model = load_voiceformerav_weights(model, paths.voiceformerav)
        return model

    if 'voiceformerav23' == init_from:
        model = load_voiceformerav_weights(model, paths.voiceformerav23)
        return model
    
    if 'voiceformer' == init_from:
        model = load_voiceformer_weights(model, paths.voiceformer)
        return model

    if '/' in init_from:
        model = load_model_weights(model, init_from)
    
    return model


def load_model_weights(model, path):
    model_states = model.state_dict()
    ckpt = torch.load(path, map_location=lambda storage, loc: storage)
    for name, param in ckpt['state_dict'].items():
        name = name.replace('demucs.', 'net.')
        if name in model_states and param.shape == model_states[name].shape:
            model_states[name].copy_(param)
            # print('1', name)
        else:
            print('0', name, param.shape)
    return model

def load_pth_model_weights(model, path, pre='net.'):
    model_states = model.state_dict()
    ckpt = torch.load(path, map_location=lambda storage, loc: storage)
 
    for name, param in ckpt.items():
        name = pre+name
        if name in model_states and param.shape == model_states[name].shape:
            model_states[name].copy_(param)
            # print('1', name)
        else:
            print('0', name)
    return model

def load_demucs_weights(model, path, pre='net.'):
    model_states = model.state_dict()
    ckpt = torch.load(path, map_location=lambda storage, loc: storage)
    for name, param in ckpt.items():
        name = pre+name
        if name in model_states and param.shape == model_states[name].shape:
            model_states[name].copy_(param)
            # print('1', name)
        else:
            print('0', name)
    return model

def load_voiceformer_weights(model, path):
    model_states = model.state_dict()
    ckpt = torch.load(path, map_location=lambda storage, loc: storage)
    for name, param in ckpt['state_dict'].items():
        name = name.replace('demucs.', 'net.')
        if name in model_states and param.shape == model_states[name].shape:
            model_states[name].copy_(param)
            # print('1', name)
        else:
            print('0', name, param.shape)
  
    return model

def load_voiceformerav_weights(model, path):
    model_states = model.state_dict()
    ckpt = torch.load(path, map_location=lambda storage, loc: storage)
    for name, param in ckpt['state_dict'].items():
        name = name.replace('demucs.', 'net.')
        if name in model_states and param.shape == model_states[name].shape:
            model_states[name].copy_(param)
            # print('1', name)
        else:
            print('0', name, param.shape)
  
    return model

def load_tacotron2_weights(model, path):
    model_states = model.state_dict()
    ckpt = torch.load(path, map_location=lambda storage, loc: storage)
    for name, param in ckpt['state_dict'].items():
        name = 'net.'+name.replace('encoder.', 'phoneme_encoder.')
        name = name.replace('convolutions.0.0.conv.bias', 'convolutions.0.convolution1d.bias')
        name = name.replace('convolutions.0.0.conv.weight', 'convolutions.0.convolution1d.weight')
        name = name.replace('convolutions.0.1.', 'convolutions.0.batch_normalization.')
        name = name.replace('convolutions.1.0.conv.bias', 'convolutions.1.convolution1d.bias')
        name = name.replace('convolutions.1.0.conv.weight', 'convolutions.1.convolution1d.weight')
        name = name.replace('convolutions.1.1.', 'convolutions.1.batch_normalization.')
        name = name.replace('convolutions.2.0.conv.bias', 'convolutions.2.convolution1d.bias')
        name = name.replace('convolutions.2.0.conv.weight', 'convolutions.2.convolution1d.weight')
        name = name.replace('convolutions.2.1.', 'convolutions.2.batch_normalization.')
        if name in model_states and param.shape == model_states[name].shape:
            model_states[name].copy_(param)
            # print('1', name)
        else:
            print('0', name, param.shape)

    return model



def initialise_weights(model, init_type='xavier', init_gain=0.02):
    """Initialize model weights.

    Parameters:
        model (network)     -- network to be initialized
        init_type (str)     -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)   -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                 torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                 torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'kaiming_uniform':
                 torch.nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                 torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'uniform':
                 torch.nn.init.xavier_uniform_(m.weight.data)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                 torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
             torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
             torch.nn.init.constant_(m.bias.data, 0.0)
    
    model.apply(init_func)  # apply the initialization function <init_func>       