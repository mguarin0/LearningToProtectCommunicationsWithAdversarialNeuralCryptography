import torch
import torch.nn.functional as F

class MixTransformNN(torch.nn.Module):
    def __init__(self, D_in, H):

        super(MixTransformNN, self).__init__()
        self.fc_layer = torch.nn.Linear(D_in, H)
        self.conv1 = torch.nn.Conv1d(in_channels=1,
                                     out_channels=2,
                                     kernel_size=4,
                                     stride=1,
                                     padding=2)
        self.conv2 = torch.nn.Conv1d(in_channels=2,
                                     out_channels=4,
                                     kernel_size=2,
                                     stride=2)
        self.conv3 = torch.nn.Conv1d(in_channels=4,
                                     out_channels=4,
                                     kernel_size=1,
                                     stride=1)
        self.conv4 = torch.nn.Conv1d(in_channels=4,
                                     out_channels=1,
                                     kernel_size=1,
                                     stride=1)
    # end

    def forward(self, x):

        x = x[None, :, :].transpose(0, 1)

        x = F.sigmoid(self.fc_layer(x))

        x = F.sigmoid(self.conv1(x))

        x = F.sigmoid(self.conv2(x))

        x = F.sigmoid(self.conv3(x))

        x = F.tanh(self.conv4(x))

        return torch.squeeze(x)
    # end
# end
