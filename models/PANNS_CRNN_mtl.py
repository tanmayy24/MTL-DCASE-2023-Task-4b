from .PANNS import Cnn14
import torch
import torch.nn as nn
import torch.nn.functional as F

def clip_mse(output, target):
    loss_function = torch.nn.MSELoss(reduction='mean')
    loss = loss_function(output, target)
    return loss

def curriculum_loss(output, target):
    loss_function = nn.MSELoss(reduction='mean')
    loss = loss_function(output, target)
    weighted_loss = torch.mean(loss * target)  # Multiply the loss by example weights
    return weighted_loss

def kl_loss(pred_logits, soft_labels):
    # Apply softmax to predicted logits
    pred_probs = F.softmax(pred_logits, dim=1)
    
    # Apply log to probabilities
    pred_log_probs = torch.log(pred_probs)
    
    # Compute the KL divergence loss
    kl_loss = F.kl_div(pred_log_probs, soft_labels, reduction='batchmean')
    
    return kl_loss

def w_clip_mse(output, target):
    mse_loss = torch.mean((output - target)**2, dim=1)
    weighted_mse_loss = mse_loss * torch.tensor([29.58546256, 14.46457032,  5.96703687,  5.94959249,  6.2427031 ,29.62461403, 28.44515036, 28.31323777, 29.06057984, 29.14887153,29.19956522, 27.98291667, 28.02963272,28.46926664, 34.96043727,33.86737267, 34.07356672]).cuda()
    return torch.mean(weighted_mse_loss)

def focal_loss_soft_label(pred, target, gamma=2, alpha=None):
    # pred: predicted probability distribution (batch_size x num_classes)
    # target: true probability distribution (batch_size x num_classes)
    # gamma: focusing parameter (default: 2)
    # alpha: weighting parameter for each class (optional)
    
    # Compute cross-entropy loss
    ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    
    # Convert soft label targets to hard label targets
    target_class = torch.argmax(target, dim=1)
    
    # Compute weighting factor for each class
    if alpha is not None:
        alpha = alpha.to(pred.device)
        weight = torch.gather(alpha, 0, target_class.view(-1, 1)).squeeze()
    else:
        weight = 1.
    
    # Compute focal loss
    pt = torch.exp(-ce_loss)
    fl_loss = (1 - pt) ** gamma * ce_loss * weight
    
    # Average loss over the batch
    fl_loss = fl_loss.mean()
    
    return fl_loss

def soft_label_kl_loss(y_pred, y_soft):
    """
    Compute the KL divergence loss between predicted probabilities and soft label probabilities.
    """
    #y_true = torch.argmax(y_soft, dim=1) # convert soft labels to one-hot encoded labels
    log_prob = F.log_softmax(y_pred, dim=1) # apply softmax to predicted logits and take log
    kl_loss = F.kl_div(log_prob, y_soft, reduction='batchmean') # compute KL divergence loss
    return kl_loss

def soft_label_focal_loss(y_pred, y_soft, gamma=2.0, alpha=None):
    """
    Compute the Focal loss between predicted probabilities and soft label probabilities.
    """
    print(y_pred.shape)
    print(y_soft.shape)
    sys.exit()
    y_true = torch.argmax(y_soft, dim=1) # convert soft labels to one-hot encoded labels
    log_prob = F.log_softmax(y_pred, dim=1) # apply softmax to predicted logits and take log
    prob = torch.exp(log_prob) # exponentiate to get probabilities
    pt = prob.gather(1, y_true.unsqueeze(1)).squeeze() # select probabilities for true class
    fl_weight = (1 - pt)**gamma # compute focal loss weight
    if alpha is not None:
        fl_weight = alpha.gather(0, y_true) * fl_weight # apply class weighting if specified
    fl_loss = F.nll_loss(fl_weight * log_prob, y_true, reduction='mean') # compute weighted negative log-likelihood loss
    return fl_loss


class my_CRNN(nn.Module):
    def __init__(self, classes_num, cnn_filters, rnn_hid, _dropout_rate):
        super(my_CRNN, self).__init__()

        self.cnn = Cnn14(527)
        self.dropout = nn.Dropout(_dropout_rate)
        
        self.conv = nn.Conv2d(in_channels=512, 
                              out_channels=512,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)

        self.gru1 = nn.GRU(2048, rnn_hid, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(rnn_hid*2, rnn_hid)
        self.linear2 = nn.Linear(rnn_hid, classes_num)
        
        
        self.gru1_c = nn.GRU(2048, rnn_hid, bidirectional=True, batch_first=True)
        self.linear1_c = nn.Linear(rnn_hid*2, rnn_hid)
        self.linear2_c = nn.Linear(rnn_hid, 5)
        
        
        self.gru1_i = nn.GRU(2048, rnn_hid, bidirectional=True, batch_first=True)
        self.linear1_i = nn.Linear(rnn_hid*2, rnn_hid)
        self.linear2_i = nn.Linear(rnn_hid, 2)

    def forward(self, input):
        x = input[:,None,:,:]
        x = self.cnn(x)
        x = x['cnn_embedding']
        
        #x = self.conv(x)
        
        x = x.permute(0, 2, 1, 3)
        x = x.reshape((x.shape[0], x.shape[1], -1))
        
        # Bidirectional layer
        recurrent, _ = self.gru1(x)
        x_o = self.linear1(recurrent)
        x_o = self.linear2(x_o)
        
        # Bidirectional layer
        recurrent_c, _ = self.gru1_c(x)
        x_c = self.linear1_c(recurrent_c)
        x_c = self.linear2_c(x_c)
        
        # Bidirectional layer
        recurrent_i, _ = self.gru1_i(x)
        x_i = self.linear1_i(recurrent_i)
        x_i = self.linear2_i(x_i)
        return x_o, x_c, x_i
    
    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.cnn.load_state_dict(checkpoint['model'], strict=False)



