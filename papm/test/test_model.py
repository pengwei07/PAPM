from model.burgers_nets import ConvLSTM, DilResNet, CNO_model, UNet_model, fno_model, ppnn_model, percnn_model, papm_model
import torch

x = torch.rand([3,100,2,64,64])
phy = torch.rand([3,1])

model=ConvLSTM(mode='single_step')
y=model(x, phy=phy, step=50)
print(y.shape)

model=DilResNet(mode='single_step')
y=model(x, phy=phy, step=50)
print(y.shape)

model=CNO_model(mode='single_step')
y=model(x, phy=phy, step=50)
print(y.shape)

model=UNet_model(mode='single_step')
y=model(x, phy=phy, step=50)
print(y.shape)

model=fno_model(mode='single_step')
y=model(x, phy=phy, step=50)
print(y.shape)


model=ConvLSTM(mode='rollout')
y=model(x, phy=phy, step=50)
print(y.shape)

model=DilResNet(mode='rollout')
y=model(x, phy=phy, step=50)
print(y.shape)

model=CNO_model(mode='rollout')
y=model(x, phy=phy, step=50)
print(y.shape)

model=UNet_model(mode='rollout')
y=model(x, phy=phy, step=50)
print(y.shape)

model=fno_model(mode='rollout')
y=model(x, phy=phy, step=50)
print(y.shape)

model=ppnn_model()
y=model(x, phy=phy, step=50)
print(y.shape)

model=percnn_model()
y=model(x, phy=phy, step=50)
print(y.shape)

model=papm_model()
y=model(x, phy=phy, step=50)
print(y.shape)
