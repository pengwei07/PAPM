import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchdiffeq import odeint
#from torchdiffeq import odeint_adjoint as odeint
# pdeunet
from .pdeunet import FourierUnet
# cnonet
from .pdecno.cno import CNO

############# 
# convlstm
class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out

class EnhancedConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(EnhancedConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Convolution layer for the gates
        self.gates = nn.Conv2d(input_channels * 2, 4 * hidden_channels, kernel_size, padding=self.padding)
        
        # Batch normalization for the gates
        self.bn = nn.ModuleList([nn.BatchNorm2d(hidden_channels) for _ in range(4)])
        
        # Extra convolutional layers and residual block
        self.conv1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=self.padding)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=self.padding)
        self.residual_block = ResidualBlock(hidden_channels, kernel_size)
        self.activation = nn.ReLU()
        
        # end
        self.conv3 = nn.Conv2d(hidden_channels, input_channels, kernel_size, padding=self.padding)

    def forward(self, x, hidden):
        h_cur, c_cur = hidden

        # Concatenate x and hidden states
        combined = torch.cat([x, h_cur], dim=1)  
        gates = self.gates(combined)

        # Split the gates for the LSTM operations
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # Apply batch normalization
        in_gate = self.bn[0](in_gate)
        remember_gate = self.bn[1](remember_gate)
        out_gate = self.bn[2](out_gate)
        cell_gate = self.bn[3](cell_gate)

        # LSTM gate operations
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)
        cell_gate = torch.tanh(cell_gate)

        c_next = (remember_gate * c_cur) + (in_gate * cell_gate)
        h_next = out_gate * torch.tanh(c_next)

        # Additional convolution layers for increased complexity
        h_next = self.conv1(h_next)
        h_next = self.conv2(h_next)
        h_next = self.residual_block(h_next)
        h_next = self.residual_block(h_next)
        h_next = self.activation(h_next)
        
        h_next = self.conv3(h_next)
        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, input_channels=2, hidden_channels=16, kernel_size=5, mode='single_step'):
        super(ConvLSTM, self).__init__()
        self.cell = EnhancedConvLSTMCell(input_channels*5, hidden_channels, kernel_size)
        assert mode == 'single_step' or 'rollout'
        self.mode = mode

    def forward(self, x, phy=None, step=20):
        init_step = 5
        x = x[:,:init_step,...]
        b,t,c,h,w = x.size()
        x = x.permute(0,3,4,1,2).reshape((b,h,w,t*c))
        x = x.permute(0,3,1,2)
        out_state = x

        hidden = (torch.zeros(b, t*c, h, w, device=x.device), 
                  torch.zeros(b, 16, h, w, device=x.device))
        if self.mode == 'single_step':
            for i in range(step):
                # forward
                hidden = self.cell(x, hidden)
                x = hidden[0]
                out_state = torch.cat((out_state, hidden[0][:,-c:,...]), dim=1)

        if self.mode == 'rollout':
            for i in range(step//5):
                # forward
                hidden = self.cell(x, hidden)
                x = hidden[0]
                out_state = torch.cat((out_state, hidden[0][:,:,...]), dim=1)
        out_state = out_state[:,init_step*c:,...]
        out_state = out_state.reshape(b, step, c, h, w)

        return out_state
    
#res_net
class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rates):
        super(DilatedConvBlock, self).__init__()
        
        assert len(dilation_rates) == 7, "Expected 7 dilation rates"
        
        self.layers = nn.ModuleList()
        
        for rate in dilation_rates:
            self.layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=rate, dilation=rate)
            )
            in_channels = out_channels  # The output becomes the input for the next layer
        
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
            out = nn.ReLU()(out)
        return out + x  # Residual connection

class DilResNet(nn.Module):
    def __init__(self, input_channels=2, kernel_size=3, latent_size=24, block_depth=7, dilation_rates=(1,2,4,8,4,2,1), num_blocks=4, mode='single_step'):
        super(DilResNet, self).__init__()

        assert mode == 'single_step' or 'rollout'
        self.mode = mode

        self.encoder = nn.Conv2d(input_channels*5, latent_size, kernel_size, padding=1)  # Encoder

        # Processor: Series of dilated convolution blocks with residual connections
        self.processor_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.processor_blocks.append(DilatedConvBlock(latent_size, latent_size, kernel_size, dilation_rates))

        self.decoder = nn.Conv2d(latent_size, input_channels*5, kernel_size, padding=1)  # Decoder

    def forward(self, x, phy=None, step=20):
        init_step = 5
        x = x[:,:init_step,...]
        b,t,c,h,w = x.size()
        x = x.permute(0,3,4,1,2).reshape((b,h,w,t*c))
        x = x.permute(0,3,1,2)
        out_state = x

        if self.mode == 'single_step':
            for i in range(step):
                # forward
                encoded = self.encoder(x)
                # Processing
                processed = encoded
                for block in self.processor_blocks:
                    processed = block(processed)
                # Decoding
                decoded = self.decoder(processed)
                x = torch.cat((x[:,c:,...],decoded[:,-c:,...]),dim=1)
                out_state = torch.cat((out_state, decoded[:,-c:,...]), dim=1)

        if self.mode == 'rollout':
            for i in range(step//5):
                # forward
                encoded = self.encoder(x)
                # Processing
                processed = encoded
                for block in self.processor_blocks:
                    processed = block(processed)
                # Decoding
                decoded = self.decoder(processed)
                x = decoded[:,:,...]
                out_state = torch.cat((out_state, decoded[:,:,...]), dim=1)

        out_state = out_state[:,init_step*c:,...]
        out_state = out_state.reshape(b, step, c, h, w)
        
        return out_state
# fno
############# 
# fno
class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, num_channels, output_step, modes1, modes2, width, initial_step):
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        #self.padding = 0 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(initial_step*num_channels, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 64)
        self.fc21 = nn.Linear(64, output_step)
        self.fc22 = nn.Linear(64, output_step)
        self.fc23 = nn.Linear(64, output_step)

    def forward(self, x):
        # x dim = [b, x1, x2, t*v]
        batch_each = x.shape[0]
        h_dim, w_dim = x.shape[-2], x.shape[-1]
        x = x.permute(0, 3, 4, 1, 2).reshape((batch_each,h_dim,w_dim,-1))
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        c_1 = self.fc21(x)
        c_2 = self.fc22(x)
        
        xx1 = c_1.permute(0, 3, 1, 2)
        xx2 = c_2.permute(0, 3, 1, 2)
        
        return torch.stack((xx1, xx2), dim=2)
    
class fno_model(nn.Module):
    def __init__(self, num_channels=2, width=20, modes1=12, modes2=12, initial_step=5, mode='single_step'):
        super(fno_model, self).__init__()
        assert mode == 'single_step' or 'rollout'
        self.mode = mode
        if self.mode == 'single_step':
            self.fno2d = FNO2d(num_channels, 1, modes1, modes2, width, initial_step)
        if self.mode == 'rollout':
            self.fno2d = FNO2d(num_channels, 5, modes1, modes2, width, initial_step)

    def forward(self, x, phy=None, step=20):
        # x.shape = [batch, 5, 3, 160, 160]
        init_step = 5
        outputs = []
        init_state = x[:, :init_step, ...]
        if self.mode == 'single_step':
            output_step = 1
            for _ in range(step):
                phi_next = self.fno2d(init_state)
                outputs.append(phi_next)
                # 更新init_state为新的序列
                init_state = torch.cat((init_state[:, output_step:, ...], phi_next), dim=1)
        
        if self.mode == 'rollout':
            output_step = 5
            for _ in range(step//output_step):
                phi_next = self.fno2d(init_state)
                outputs.append(phi_next)
                # 更新init_state为新的序列
                init_state = phi_next
        
        # 返回整个输出序列
        output = torch.cat(outputs, dim=1)
        return output

# UNet
class UNet_model(nn.Module):
    def __init__(self, input_channels=2, mode='single_step'):
        super(UNet_model, self).__init__()
        assert mode == 'single_step' or 'rollout'
        self.mode = mode
        if self.mode == 'single_step':
            self.model = FourierUnet(n_input_scalar_components=input_channels, n_output_scalar_components=input_channels,time_history=5,time_future=1)
        if self.mode == 'rollout':
            self.model = FourierUnet(n_input_scalar_components=input_channels, n_output_scalar_components=input_channels,time_history=5,time_future=5)
    def forward(self, x, phy=None, step=20):
        init_step = 5
        x = x[:,:init_step,...]
        if self.mode == 'single_step':
            num_steps = step
            out_state = x
            out_state = self.model(x, num_steps)

        if self.mode == 'rollout':
            num_steps = step // 5
            out_state = x
            out_state = self.model(x, num_steps)
        return out_state
    
# CNO
class CNO_model(nn.Module):
    def __init__(self, input_channels=2, mode='single_step'):
        super(CNO_model, self).__init__()
        assert mode == 'single_step' or 'rollout'
        self.mode = mode
        if self.mode == 'single_step':
            self.net = CNO(in_dim=input_channels*5,in_size=64,out_dim=input_channels)
        if self.mode == 'rollout':
            self.net = CNO(in_dim=input_channels*5,in_size=64,out_dim=input_channels*5)

    def forward(self, x, phy=None, step=20):
        init_step = 5
        x = x[:,:init_step,...]
        b,t,c,h,w = x.size()
        x = x.reshape((b,t*c,h,w))
        out_state = x

        if self.mode == 'single_step':
            for i in range(step):
                # forward
                out = self.net(x)
                x = torch.cat((x[:,c:,...], out), dim=1)
                out_state = torch.cat((out_state, out), dim=1)


        if self.mode == 'rollout':
            for i in range(step//5):
                 # forward
                out = self.net(x)
                x = out
                out_state = torch.cat((out_state, out), dim=1)

        out_state = out_state[:,init_step*c:,...]
        out_state = out_state.reshape(b, step, c, h, w)
        return out_state
############# 
# percnn

class percnn_diffusive_flows(nn.Module):
    def __init__(self):
        super(percnn_diffusive_flows, self).__init__()
        self.input_stride = 1
        self.initial_step = 1
        self.num_channels = 1
        
        self.input_channels = self.initial_step * self.num_channels
        self.hidden_channels = 16
        
        self.Wh1_u = nn.Conv2d(in_channels=self.input_channels, out_channels=self.hidden_channels, kernel_size=1, stride=self.input_stride, padding=0, bias=True)
        self.Wh2_u = nn.Conv2d(in_channels=self.input_channels, out_channels=self.hidden_channels, kernel_size=1, stride=self.input_stride, padding=0, bias=True)
        self.Wh3_u = nn.Conv2d(in_channels=self.input_channels, out_channels=self.hidden_channels, kernel_size=1, stride=self.input_stride, padding=0, bias=True)
        self.Wh4_u = nn.Conv2d(in_channels=self.hidden_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.Wh1_v = nn.Conv2d(in_channels=self.input_channels, out_channels=self.hidden_channels, kernel_size=1, stride=self.input_stride, padding=0, bias=True)
        self.Wh2_v = nn.Conv2d(in_channels=self.input_channels, out_channels=self.hidden_channels, kernel_size=1, stride=self.input_stride, padding=0, bias=True)
        self.Wh3_v = nn.Conv2d(in_channels=self.input_channels, out_channels=self.hidden_channels, kernel_size=1, stride=self.input_stride, padding=0, bias=True)
        self.Wh4_v = nn.Conv2d(in_channels=self.hidden_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        
        #self.filter_list = [self.Wh1_u, self.Wh2_u, self.Wh3_u, self.Wh4_u, self.Wh1_v, self.Wh2_v, self.Wh3_v, self.Wh4_v]
        #self.init_filter(self.filter_list, c=0.02)
        
    def init_filter(self, filter_list, c):
        '''
        :param filter_list: list of filter for initialization
        :param c: constant multiplied on Xavier initialization
        '''
        for filter in filter_list:
            # Xavier initialization and then scale
            torch.nn.init.xavier_uniform_(filter.weight)
            filter.weight.data = c*filter.weight.data
            # filter.weight.data.uniform_(-c * np.sqrt(1 / (5 * 5 * 16)), c * np.sqrt(1 / (5 * 5 * 16)))
            if filter.bias is not None:
                filter.bias.data.fill_(0.0)
                
    def forward(self, x, nu):
        u, v = x[:,0:1,:,:], x[:,1:2,:,:]
        nu = nu.unsqueeze(-1).unsqueeze(-1).expand_as(x[:,0:1,:,:]).to(x.device)
        # v = v.unsqueeze(-1).unsqueeze(-1).expand_as(x[:,0:1,:,:]).to(device)
        u_diff = (nu) * self.Wh4_u(self.Wh1_u(u)*self.Wh2_u(u)*self.Wh3_u(u))
        v_diff = (nu) * self.Wh4_v(self.Wh1_v(v)*self.Wh2_v(v)*self.Wh3_v(v))
        diffusive = torch.cat((u_diff, v_diff), dim=1)
        return diffusive

class percnn_convective_flow(nn.Module):
    def __init__(self):
        super(percnn_convective_flow, self).__init__()
        self.input_stride = 1
        self.initial_step = 1
        self.num_channels = 1
        
        self.input_channels = self.initial_step * self.num_channels
        self.hidden_channels = 16
        
        self.Wh1_ux = nn.Conv2d(in_channels=self.input_channels, out_channels=self.hidden_channels, kernel_size=1, stride=self.input_stride, padding=0, bias=True)
        self.Wh2_ux = nn.Conv2d(in_channels=self.input_channels, out_channels=self.hidden_channels, kernel_size=1, stride=self.input_stride, padding=0, bias=True)
        self.Wh3_ux = nn.Conv2d(in_channels=self.hidden_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.Wh1_uy = nn.Conv2d(in_channels=self.input_channels, out_channels=self.hidden_channels, kernel_size=1, stride=self.input_stride, padding=0, bias=True)
        self.Wh2_uy = nn.Conv2d(in_channels=self.input_channels, out_channels=self.hidden_channels, kernel_size=1, stride=self.input_stride, padding=0, bias=True)
        self.Wh3_uy = nn.Conv2d(in_channels=self.hidden_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.Wh1_vx = nn.Conv2d(in_channels=self.input_channels, out_channels=self.hidden_channels, kernel_size=1, stride=self.input_stride, padding=0, bias=True)
        self.Wh2_vx = nn.Conv2d(in_channels=self.input_channels, out_channels=self.hidden_channels, kernel_size=1, stride=self.input_stride, padding=0, bias=True)
        self.Wh3_vx = nn.Conv2d(in_channels=self.hidden_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.Wh1_vy = nn.Conv2d(in_channels=self.input_channels, out_channels=self.hidden_channels, kernel_size=1, stride=self.input_stride, padding=0, bias=True)
        self.Wh2_vy = nn.Conv2d(in_channels=self.input_channels, out_channels=self.hidden_channels, kernel_size=1, stride=self.input_stride, padding=0, bias=True)
        self.Wh3_vy = nn.Conv2d(in_channels=self.hidden_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        
        #self.filter_list = [self.Wh1_ux,self.Wh2_ux,self.Wh3_ux,self.Wh1_uy,self.Wh2_uy,self.Wh3_uy,self.Wh1_vx,self.Wh2_vx,self.Wh3_vx,self.Wh1_vy,self.Wh2_vy,self.Wh3_vy]
        #self.init_filter(self.filter_list, c=0.02)
        
    def init_filter(self, filter_list, c):
        '''
        :param filter_list: list of filter for initialization
        :param c: constant multiplied on Xavier initialization
        '''
        for filter in filter_list:
            # Xavier initialization and then scale
            torch.nn.init.xavier_uniform_(filter.weight)
            filter.weight.data = c*filter.weight.data
            # filter.weight.data.uniform_(-c * np.sqrt(1 / (5 * 5 * 16)), c * np.sqrt(1 / (5 * 5 * 16)))
            if filter.bias is not None:
                filter.bias.data.fill_(0.0)
        
    def forward(self, x):
        u, v = x[:,0:1,:,:], x[:,1:2,:,:]
        u_grad_x = self.Wh3_ux(self.Wh1_ux(u)*self.Wh2_ux(u))
        u_grad_y = self.Wh3_uy(self.Wh1_uy(u)*self.Wh2_uy(u))
        v_grad_x = self.Wh3_vx(self.Wh1_vx(v)*self.Wh2_vx(v))
        v_grad_y = self.Wh3_vy(self.Wh1_vy(v)*self.Wh2_vy(v))
        # convective flow
        convective_u = u * u_grad_x + v * u_grad_y
        convective_v = u * v_grad_x + v * v_grad_y
        convective = torch.cat((convective_u, convective_v), dim=1)
        return convective
    
class percnn_source(nn.Module):
    def __init__(self):
        super(percnn_source, self).__init__()
        # Nonlinear term for u (up to 3rd order)
        self.input_stride = 1
        self.input_channels = 2
        self.hidden_channels = 16
        
        self.Wh1_s = nn.Conv2d(in_channels=self.input_channels, out_channels=self.hidden_channels, kernel_size=1, stride=self.input_stride, padding=0, bias=True)
        self.Wh2_s = nn.Conv2d(in_channels=self.input_channels, out_channels=self.hidden_channels, kernel_size=1, stride=self.input_stride, padding=0, bias=True)
        self.Wh3_s = nn.Conv2d(in_channels=self.input_channels, out_channels=self.hidden_channels, kernel_size=1, stride=self.input_stride, padding=0, bias=True)
        self.Wh4_s = nn.Conv2d(in_channels=self.hidden_channels, out_channels=2, kernel_size=1, stride=1, padding=0, bias=True)
        
        # initialize filter's wweight and bias
        self.filter_list = [self.Wh1_s, self.Wh2_s, self.Wh3_s, self.Wh4_s]
        self.init_filter(self.filter_list, c=0.02)
    
    def init_filter(self, filter_list, c):
        '''
        :param filter_list: list of filter for initialization
        :param c: constant multiplied on Xavier initialization
        '''
        for filter in filter_list:
            # Xavier initialization and then scale
            torch.nn.init.xavier_uniform_(filter.weight)
            filter.weight.data = c*filter.weight.data
            # filter.weight.data.uniform_(-c * np.sqrt(1 / (5 * 5 * 16)), c * np.sqrt(1 / (5 * 5 * 16)))
            if filter.bias is not None:
                filter.bias.data.fill_(0.0)
    
    def forward(self, x):
        # x dim = [b, 2, 64, 64]
        source = self.Wh4_s(self.Wh1_s(x)*self.Wh2_s(x)*self.Wh3_s(x))
        return source

class percnn_time_stepping(nn.Module):
    def __init__(self):
        super(percnn_time_stepping, self).__init__()
        self.diffusive_flow = percnn_diffusive_flows()
        self.convective_flow = percnn_convective_flow()
        self.source = percnn_source()
        self.dt = 0.01
        
    def forward(self, x, nu):
        # stage 1
        x_end_uv = x[:,-1,:,...]
        phi_diff = self.diffusive_flow(x_end_uv, nu)
        phi_conv = self.convective_flow(x_end_uv)
        phi_source = self.source(x_end_uv)
        #print(phi_source.shape)
        uv_end = x_end_uv + (phi_diff - phi_conv + phi_source) * self.dt
        
        return uv_end
    
class percnn_model(nn.Module):
    def __init__(self):
        super(percnn_model, self).__init__()
        self.time_stepping = percnn_time_stepping()

    def forward(self, x, phy=None, step=20):
        # x.shape = [batch, 5, 3, 256, 64]
        init_step = 5
        outputs = []
        init_state = x[:, :init_step, ...]
        for _ in range(step):
            # forward
            phi_next = self.time_stepping(init_state, phy)
            outputs.append(phi_next)
            init_state = torch.cat((init_state[:, 1:, ...], phi_next.unsqueeze(1)), dim=1)
        
        # return all timesteps
        output = torch.stack(outputs, dim=1)
        return output
    
############# 
# ppnn

lap_2d_op_ppnn = [[[[    0,   -1,   0],
               [   -1,    4,   1],
               [    0,   -1,   0]]]]

lap_2d_x_ppnn = [[[[    0,   -1,   0],
               [   -1,   2,   0],
               [    0,   0,   0]]]]

class ppnn_diffusive_flows(nn.Module):
    def __init__(self):
        super(ppnn_diffusive_flows, self).__init__()
        self.dx = 2 * torch.pi / 16 # dx
        self.input_kernel_size = 3
        self.input_stride = 1
        
        self.W_laplace1 = nn.Conv2d(1, 1, self.input_kernel_size, self.input_stride, padding=1, bias=False)
        self.W_laplace2 = nn.Conv2d(1, 1, self.input_kernel_size, self.input_stride, padding=1, bias=False)
        
        weight_ori = (1/self.dx**2)*torch.tensor(lap_2d_op_ppnn, dtype=torch.float32)
        self.W_laplace1.weight = nn.Parameter(weight_ori)
        self.W_laplace2.weight = nn.Parameter(weight_ori)

        self.W_laplace1.weight.requires_grad = False
        self.W_laplace2.weight.requires_grad = False
        
    def forward(self, x, coff):
        x_pad = x
        u, v = x_pad[:,0:1,:,:], x_pad[:,1:2,:,:]
        coff_expanded = coff.unsqueeze(-1).unsqueeze(-1).expand_as(u).to(x.device)
        # print(coff_expanded.shape, u.shape, self.W_laplace1.weight.shape, self.W_laplace1(u).shape)
        u_diff = coff_expanded * self.W_laplace1(u)
        v_diff = coff_expanded * self.W_laplace2(v)
        diffusive = torch.cat((u_diff, v_diff), dim=1)
        return diffusive
        
class ppnn_convective_flow(nn.Module):
    def __init__(self):
        super(ppnn_convective_flow, self).__init__()
        self.dx = 2 * torch.pi / 16 # dx
        self.input_kernel_size = 3
        self.input_stride = 1
        
        self.W_grad_x = nn.Conv2d(1, 1, self.input_kernel_size, self.input_stride, padding=1, bias=False)
        self.W_grad_y = nn.Conv2d(1, 1, self.input_kernel_size, self.input_stride, padding=1, bias=False)
        
        weight_x = (1/self.dx)*torch.tensor(lap_2d_x_ppnn, dtype=torch.float32)
        weight_y = (1/self.dx)*torch.tensor(lap_2d_x_ppnn, dtype=torch.float32)
        
        self.W_grad_x.weight = nn.Parameter(weight_x)
        self.W_grad_y.weight = nn.Parameter(weight_y)

        self.W_grad_x.weight.requires_grad = False
        self.W_grad_y.weight.requires_grad = False
    
    def forward(self, x):
        x_pad = x
        u, v = x_pad[:,0:1,:,:], x_pad[:,1:2,:,:]
        u_grad_x = self.W_grad_x(u)
        u_grad_y = self.W_grad_y(u)
        v_grad_x = self.W_grad_x(v)
        v_grad_y = self.W_grad_y(v)
        # convective flow
        convective_u = u * u_grad_x + v * u_grad_y
        convective_v = u * v_grad_x + v * v_grad_y
        convective = torch.cat((convective_u, convective_v), dim=1)
        return convective
    
class PDE_pre(nn.Module):
    def __init__(self):
        super(PDE_pre, self).__init__()
        self.diffusive_flow = ppnn_diffusive_flows()
        self.convective_flow = ppnn_convective_flow()
    def forward(self, x, nu):
        # diffusive flows
        phi_diff = self.diffusive_flow(x,nu)
        phi_conv = self.convective_flow(x)
        w_next = phi_diff - phi_conv
        return w_next

class ppnn(nn.Module):
    def __init__(self):
        super(ppnn, self).__init__()
        self.input_c = 2
        self.init_step = 5
        self.dt = 0.01
        self.input_channels = self.input_c*self.init_step
        
        # PDE-preserving Layers
        self.pde_preserve = PDE_pre()

        # Encoding Layers
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channels + self.input_c, 48, kernel_size=6, padding=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=6, padding=2, stride=2)
        )

        # Trainable Layers
        self.trainable = nn.ModuleList([nn.Conv2d(48, 48, kernel_size=5, padding=2) for _ in range(3)])

        # Decoding Layers
        self.decoder = nn.Sequential(
            nn.Conv2d(48, 48*4*4, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.PixelShuffle(upscale_factor=4),
            nn.Conv2d(48, self.input_c, kernel_size=5, padding=2)
        )

    def forward(self, x, nu):
        batch, timesteps, c, height, width = x.size()
        x_end = x[:,-1,...]
        
        # PDEpreserving 
        pde_input = F.interpolate(x_end, scale_factor=1/4, mode='bilinear', align_corners=False)
        pde_out = self.pde_preserve(pde_input, nu)
        upsampled_pde_out = nn.functional.interpolate(pde_out, scale_factor=4, mode='bilinear', align_corners=True)
        
        # Encoding
        input_part = torch.cat((x, upsampled_pde_out.unsqueeze(1)), dim=1)
        input_part = input_part.permute(0, 3, 4, 1, 2).reshape((batch,64,64,-1))
        input_part = input_part.permute(0, 3, 1, 2)
        encoded = self.encoder(input_part) 
        for layer in self.trainable:
            train_out = layer(encoded)
        # Decoding
        decoded = self.decoder(train_out)
            
        # update
        current_state = x_end + (decoded+upsampled_pde_out) * self.dt

        return current_state
    
class ppnn_model(nn.Module):
    def __init__(self):
        super(ppnn_model, self).__init__()
        self.PPNN = ppnn()

    def forward(self, x, phy=None, step=20):
        # x.shape = [batch, 5, 2, 64, 64]
        init_step = 5
        outputs = []
        init_state = x[:, :init_step, ...]
        for _ in range(step):
            # forward
            phi_next = self.PPNN(init_state, phy)
            outputs.append(phi_next)
            init_state = torch.cat((init_state[:, 1:, ...], phi_next.unsqueeze(1)), dim=1)
        
        # return all timesteps
        output = torch.stack(outputs, dim=1)
        return output
    
############# 
# papm
Nx = 64
Ny = 64
X = np.linspace(0, 2 * np.pi, Nx, endpoint=False)
Y = np.linspace(0, 2 * np.pi, Ny, endpoint=False)
X, Y = np.meshgrid(X, Y)

lap_2d_op = [[[[    0,   0, -1/12,   0,     0],
               [    0,   0,   4/3,   0,     0],
               [-1/12, 4/3,   - 5, 4/3, -1/12],
               [    0,   0,   4/3,   0,     0],
               [    0,   0, -1/12,   0,     0]]]]

lap_2d_x = [[[[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [1/12, -8/12, 0, 8/12, -1/12],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]]]]

lap_2d_y = [[[[0, 0, 1/12, 0, 0],
            [0, 0, -8/12, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 8/12, 0, 0],
            [0, 0, -1/12, 0, 0]]]]

class papm_diffusive_flows(nn.Module):
    def __init__(self):
        super(papm_diffusive_flows, self).__init__()
        self.dx = 2 * torch.pi / 64 # dx
        self.input_kernel_size = 5
        self.input_stride = 1
        
        self.W_laplace1 = nn.Conv2d(1, 1, self.input_kernel_size, self.input_stride, padding=0, bias=False)
        self.W_laplace2 = nn.Conv2d(1, 1, self.input_kernel_size, self.input_stride, padding=0, bias=False)
        
        weight_ori = (1/self.dx**2)*torch.tensor(lap_2d_op, dtype=torch.float32)
        self.W_laplace1.weight = nn.Parameter(weight_ori)
        self.W_laplace2.weight = nn.Parameter(weight_ori)
        
    def periodic_padding(self, x, kernel_size):
        pad = kernel_size // 2
        # Pad horizontally
        left_pad = x[:, :, :, -pad:]
        right_pad = x[:, :, :, :pad]
        x = torch.cat([left_pad, x, right_pad], dim=3)
        # Pad vertically
        up_pad = x[:, :, -pad:, :]
        down_pad = x[:, :, :pad, :]
        x = torch.cat([up_pad, x, down_pad], dim=2)
        return x
    
    def forward(self, x, coff):
        x_pad = self.periodic_padding(x, 5)
        u, v = x_pad[:,0:1,:,:], x_pad[:,1:2,:,:]
        coff_expanded = coff.unsqueeze(-1).unsqueeze(-1).expand_as(x[:,0:1,:,:]).to(x.device)
        u_diff = coff_expanded * self.W_laplace1(u)
        v_diff = coff_expanded * self.W_laplace2(v)
        diffusive = torch.cat((u_diff, v_diff), dim=1)
        return diffusive
        
class papm_convective_flow(nn.Module):
    def __init__(self):
        super(papm_convective_flow, self).__init__()
        self.dx = 2 * torch.pi / 64 # dx
        self.input_kernel_size = 5
        self.input_stride = 1
        
        self.W_grad_x = nn.Conv2d(1, 1, self.input_kernel_size, self.input_stride, padding=0, bias=False)
        self.W_grad_y = nn.Conv2d(1, 1, self.input_kernel_size, self.input_stride, padding=0, bias=False)
        
        weight_x = (1/self.dx)*torch.tensor(lap_2d_x, dtype=torch.float32)
        weight_y = (1/self.dx)*torch.tensor(lap_2d_y, dtype=torch.float32)
        
        self.W_grad_x.weight = nn.Parameter(weight_x)
        self.W_grad_y.weight = nn.Parameter(weight_y)
        #self.relu = nn.ReLU()
        
    def periodic_padding(self, x, kernel_size):
        pad = kernel_size // 2
        # Pad horizontally
        left_pad = x[:, :, :, -pad:]
        right_pad = x[:, :, :, :pad]
        x = torch.cat([left_pad, x, right_pad], dim=3)
        # Pad vertically
        up_pad = x[:, :, -pad:, :]
        down_pad = x[:, :, :pad, :]
        x = torch.cat([up_pad, x, down_pad], dim=2)
        return x
    
    def forward(self, x):
        x_pad = self.periodic_padding(x, 5)
        u, v = x_pad[:,0:1,:,:], x_pad[:,1:2,:,:]
        u_grad_x = self.W_grad_x(u)
        u_grad_y = self.W_grad_y(u)
        v_grad_x = self.W_grad_x(v)
        v_grad_y = self.W_grad_y(v)
        # convective flow
        convective_u = u[:,:,2:-2,2:-2] * u_grad_x + v[:,:,2:-2,2:-2] * u_grad_y
        convective_v = u[:,:,2:-2,2:-2] * v_grad_x + v[:,:,2:-2,2:-2] * v_grad_y
        convective = torch.cat((convective_u, convective_v), dim=1)
        return convective
    
class papm_source(nn.Module):
    def __init__(self):
        super(papm_source, self).__init__()
        self.xx = torch.tensor(X, dtype=torch.float32)
        self.yy = torch.tensor(Y, dtype=torch.float32)
        self.initial_step = 5
        self.num_channels = 2
        self.width = 16
        self.conv0 = nn.Conv2d(self.initial_step*self.num_channels + 2, self.width, 5, 1, padding=0, bias=True)
        self.conv1 = nn.Conv2d(self.width, self.width, 5, 1, padding=2, bias=True)
        self.conv2 = nn.Conv2d(self.width, self.width, 3, 1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(self.width, self.num_channels, 3, 1, padding=1, bias=False)
        
        # initialize filter's wweight and bias
        self.filter_list = [self.conv0, self.conv1, self.conv2, self.conv3]
        self.init_filter(self.filter_list, c=0.02)
    
    def init_filter(self, filter_list, c):
        '''
        :param filter_list: list of filter for initialization
        :param c: constant multiplied on Xavier initialization
        '''
        for filter in filter_list:
            # Xavier initialization and then scale
            torch.nn.init.xavier_uniform_(filter.weight)
            filter.weight.data = c*filter.weight.data
            # filter.weight.data.uniform_(-c * np.sqrt(1 / (5 * 5 * 16)), c * np.sqrt(1 / (5 * 5 * 16)))
            if filter.bias is not None:
                filter.bias.data.fill_(0.0)

    def periodic_padding(self, x, kernel_size):
        pad = kernel_size // 2
        # Pad horizontally
        left_pad = x[:, :, :, -pad:]
        right_pad = x[:, :, :, :pad]
        x = torch.cat([left_pad, x, right_pad], dim=3)
        # Pad vertically
        up_pad = x[:, :, -pad:, :]
        down_pad = x[:, :, :pad, :]
        x = torch.cat([up_pad, x, down_pad], dim=2)
        return x
    
    def forward(self, x):
        # 
        batch_each = x.shape[0]
        x = x.permute(0, 3, 4, 1, 2).reshape((batch_each,64,64,-1))
        x = x.permute(0, 3, 1, 2)
        # cat
        all = torch.cat((self.xx.unsqueeze(0).unsqueeze(0).repeat(batch_each, 1, 1, 1).to(x.device), self.yy.unsqueeze(0).unsqueeze(0).repeat(batch_each, 1, 1, 1).to(x.device), x), dim=1)
        # 
        x_pad = self.periodic_padding(all, 5)
        x = F.gelu(self.conv0(x_pad))
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = self.conv3(x)
        return x

class papm_model(nn.Module):
    def __init__(self, stepping_way = 0):
        super(papm_model, self).__init__()
        self.stepping = stepping_way # 1 for Neural ODE, or 0 for Euler
        self.diffusive_flow = papm_diffusive_flows()
        self.convective_flow = papm_convective_flow()
        self.source = papm_source()
        self.dt = 0.01

    def forward(self, x, phy=None, step=20):
        init_step = 5
        outputs = []
        init_state = x[:, :init_step, ...]
        for _ in range(step):
            x_end = init_state[:,-1,...]
            phi_diff = self.diffusive_flow(x_end,phy)
            phi_conv = self.convective_flow(x_end)
            phi_source = self.source(init_state)
            delta =  phi_diff - phi_conv + phi_source
            if self.stepping == 0:
                phi_next = x_end + delta * self.dt
            elif self.stepping == 1:
                phi_next = odeint(delta, x_end, self.dt)
            outputs.append(phi_next)
            init_state = torch.cat((init_state[:, 1:, ...], phi_next.unsqueeze(1)), dim=1)
            
        output = torch.stack(outputs, dim=1)
        return output