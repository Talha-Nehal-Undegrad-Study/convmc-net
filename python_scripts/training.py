import torch

try:
    import convmc
except ImportError:
    print("[INFO] Cloning the repository and importing convmc & dataset_preprocessing script...")
    subprocess.run(["git", "clone", "https://github.com/TalhaAhmed2000/convmc-net.git"])
    subprocess.run(["mv", "convmc-net/python_scripts", "py_scripts"])
    sys.path.append('py_scripts')
    import convmc

from convmc import to_var, UnfoldedNet3dC_admm, UnfoldedNet2dC_convmc

# Defining a function which sets up the default state of the parameters before training
def get_default_param(hyper_param_net, gpu = True):
    params_net = {}
    if hyper_param_net['Model'] == 'ADMM-Net':
        params_net['layers'] = 5
        
        params_net['initial_neta'] = 1.81    # fixed
        params_net['initial_lamda1'] = 0.051 # fixed
        params_net['initial_lamda2'] = 0.049 # fixed
        params_net['initial_v'] = 0
        
        params_net['initial_S'] = 0.05001 #fixed
        
        params_net['initial_P'] = 0.2401 #fixed
        
        params_net['coef_gamma'] = 0.4001
        
        params_net['CalInGPU'] = gpu #whether to calculate in GPU
        params_net['size1'] = 49
        params_net['size2'] = 60
    
    else:
        params_net['layers'] = 5 #1 #2 #3 #4 #5 #6 #7 #8 #9 #10
        params_net['kernel'] = [(3, 1)] * 3 + [(3, 1)] * 7
        params_net['initial_mu_inverse'] = 0.0
        
        params_net['initial_y1']= 0.8
        
        params_net['coef_mu_inverse'] = 0.36
        
        params_net['CalInGPU'] = gpu # whether to calculate in GPU
        params_net['kernel'] = params_net['kernel'][0:params_net['layers']]
        params_net['size1'] = 49
        params_net['size2'] = 60
        
    return params_net

def get_hyperparameter_grid(Model, TrainInstances, ValInstances, BatchSize, ValBatchSize, Alpha, num_epochs, learning_rate):
  hyper_param = {}

  hyper_param['Model'] = Model
  hyper_param['TrainInstances'] = TrainInstances
  hyper_param['ValInstances'] = ValInstances
  hyper_param['BatchSize'] = BatchSize
  hyper_param['ValBatchSize'] = ValBatchSize
  hyper_param['Alpha'] = Alpha
  hyper_param['Epochs'] = num_epochs
  hyper_param['Lr'] = learning_rate

  return hyper_param

# Defining a fucntion for loading/instantiating the model based on some boolean values and updates log
def get_model(params_net, hyper_param_net, log, path_whole = None, path_dict = None, loadmodel = False, load_frm = None):
    # Construct Model
    print('Configuring Network...\n')
    log.write('Configuring network...\n')

    if not loadmodel:
        print('Instantiating Model...\n')
        log.write('Instantiating Model...\n')
        if hyper_param_net['Model'] == 'ConvMC-Net':
            net = UnfoldedNet2dC_convmc(params_net)
            print('Model Instantiated...\n')
            log.write('Model Instantiated...\n')
        else:
            net = UnfoldedNet3dC_admm(params_net)
            print('Model Instantiated...\n')
            log.write('Model Instantiated...\n')

    else:
        print('Loading Model...\n')
        log.write('Loading Model...\n')
        if hyper_param_net['Model'] == 'ConvMC-Net':
            if load_frm == 'state_dict':
                net = UnfoldedNet2dC_convmc(params_net)
                state_dict = torch.load(path_dict, map_location = 'cpu')
                net.load_state_dict(state_dict)
                print('Model loaded from state dict...\n')
                log.write('Model loaded from state dict...\n')
            elif load_frm == 'whole_model':
                net = torch.load(path_whole)
                print('Whole model loaded...\n')
                log.write('Whole model loaded...\n')
            net.eval()
        else:
            if load_frm == 'state_dict':
                net = UnfoldedNet3dC_admm(params_net)
                state_dict = torch.load(path_dict, map_location = 'cpu')
                net.load_state_dict(state_dict)
                print('Model loaded from state dict...\n')
                log.write('Model loaded from state dict...\n')
            elif load_frm == 'whole_model':
                net = torch.load(path_whole)
                print('Whole model loaded...\n')
                log.write('Whole model loaded...\n')
            net.eval()
    net = net.cuda()

    return net

# Training functions and Plotting Functions

def train_step(model, dataloader, loss_fn, optimizer, log, CalInGPU, Alpha, TrainInstances, batch):
  # Put model in train mode
  model.train()

  # Initalize loss for lowrank matrices which will be calculated per batch for each epoch
  loss_mean = 0
  loss_lowrank_mean = 0

  for _, (D, L) in enumerate(dataloader):
    # set the gradients to zero at the beginning of each epoch
    optimizer.zero_grad()
    with torch.autograd.set_detect_anomaly(False):
      for ii in range(batch):
          inputs1 = to_var(D[ii], CalInGPU)
          targets_L = to_var(L[ii], CalInGPU)
          # Forward + backward + loss
          lst_1 = model([inputs1])
          outputs_L = lst_1[0][1]
          # Current loss
          loss = (Alpha * loss_fn(outputs_L, targets_L))/torch.square(torch.norm(targets_L, p = 'fro'))
          loss_lowrank = (loss_fn(outputs_L,targets_L))/torch.square(torch.norm(targets_L, p = 'fro'))
          loss_mean += loss.item()
          loss_lowrank_mean += loss_lowrank.item()
          loss.backward()
    optimizer.step()
  loss_mean = loss_mean/TrainInstances
  loss_lowrank_mean = loss_lowrank_mean/TrainInstances

  return loss_mean, loss_lowrank_mean

def test_step(model, dataloader, loss_fn, optimizer, log, CalInGPU, Alpha, ValInstances, batch):

  model.eval()
  loss_val_mean, loss_val_lowrank_mean = 0, 0

  # Validation
  with torch.no_grad():
    for _, (Dv, Lv) in enumerate(dataloader):
      for jj in range(batch):
        inputsv1 = to_var(Dv[jj], CalInGPU)   # "jj"th picture
        targets_Lv = to_var(Lv[jj], CalInGPU)
        lst_2 = model([inputsv1])  # Forward
        outputs_Lv = lst_2[0][1]
        # Current loss
        loss_val = Alpha * loss_fn(outputs_Lv, targets_Lv)/torch.square(torch.norm(targets_Lv, p = 'fro'))
        loss_val_lowrank = loss_fn(outputs_Lv, targets_Lv)/torch.square(torch.norm(targets_Lv, p = 'fro'))
        loss_val_mean += loss_val.item()
        loss_val_lowrank_mean += loss_val_lowrank.item()

  loss_val_mean = loss_val_mean/ValInstances
  loss_val_lowrank_mean = loss_val_lowrank_mean/ValInstances

  return loss_val_mean, loss_val_lowrank_mean
