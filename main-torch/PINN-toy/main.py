import torch
import torch.nn as nn
import numpy as np
import os

device = torch.device("cuda:0")

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 5),
            nn.Tanh(),
            nn.Linear(5, 5),
            nn.Tanh(),
            nn.Linear(5, 5),
            nn.Tanh(),
            nn.Linear(5, 5),
            nn.Tanh(),
            nn.Linear(5, 5),
            nn.Tanh(),
            nn.Linear(5, 1)
        )

    def forward(self, x, t):
        inputs = torch.cat([x,t],axis=1) # combined two arrays of 1 columns each to one array of 2 columns
        output = self.net(inputs)
        return output

def true(x, t):
    u = 6*np.exp(-3*x-2*t)
    return u

def f(x, t, net):
    u = net(x,t) # the dependent variable u is given by the network based on independent variables x,t
    ## Based on our f = du/dx - 2du/dt - u, we need du/dx and du/dt
    u = net(x,t)
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    pde = u_x - 2*u_t - u
    return pde

if __name__ == "__main__":
    torch.random.manual_seed(123)
    np.random.seed(123)

    net = Net().to(device)
    
    mse = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())
    
    ## Data from Boundary Conditions
    # u(x,0)=6e^(-3x)
    ## BC just gives us datapoints for training

    # BC tells us that for any x in range[0,2] and time=0, the value of u is given by 6e^(-3x)
    # Take say 500 random numbers of x
    x_bc = torch.autograd.Variable(torch.from_numpy(np.random.uniform(low=0.0, high=2.0, size=(500, 1))).float(), requires_grad=False).to(device)
    t_bc = torch.autograd.Variable(torch.zeros((500, 1)).float(), requires_grad=False).to(device)
    all_zeros = torch.autograd.Variable(torch.zeros((500,1)).float(), requires_grad=False).to(device)

    # compute u based on BC
    u_bc = torch.from_numpy(6*np.exp(-3*x_bc.clone().cpu().numpy())).float().to(device)

    iterations = 10000
    previous_validation_loss = 9e10

    save_dir = "output"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(iterations):
        optimizer.zero_grad()

        pt_x_bc = x_bc.clone()
        pt_t_bc = t_bc.clone()
        pt_u_bc = u_bc.clone()

        # BC loss: predict u
        net_bc_out = net(pt_x_bc, pt_t_bc)
        mse_u = mse(net_bc_out, pt_u_bc)

        # PDE loss
        x_collocation = np.random.uniform(low=0.0, high=2.0, size=(500,1))
        t_collocation = np.random.uniform(low=0.0, high=1.0, size=(500,1))

        pt_x_collocation = torch.autograd.Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
        pt_t_collocation = torch.autograd.Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
        
        f_out = f(pt_x_collocation, pt_t_collocation, net)
        mse_f = mse(f_out, all_zeros)

        # Combining the loss functions
        loss = mse_u + mse_f
        
        loss.backward() # This is for computing gradients using backward propagation
        optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta
            
        if epoch % 100 == 0:
            with torch.autograd.no_grad():
                print(epoch,"Traning Loss:",loss.data)

        if (epoch < 2000 and epoch % 250 == 0) or epoch == iterations-1:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            x=np.arange(0,2,0.02)
            t=np.arange(0,1,0.02)
            ms_x, ms_t = np.meshgrid(x, t)

            ## Just because meshgrid is used, we need to do the following adjustment
            x = np.ravel(ms_x).reshape(-1,1)
            t = np.ravel(ms_t).reshape(-1,1)
            pt_x = torch.autograd.Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
            pt_t = torch.autograd.Variable(torch.from_numpy(t).float(), requires_grad=True).to(device)

            pt_u = net(pt_x,pt_t)
            u = pt_u.data.cpu().numpy()
            u_true = true(x, t)
            ms_u = u.reshape(ms_x.shape)
            u_true = u_true.reshape(ms_x.shape)

            surf = ax.plot_surface(ms_x,ms_t,ms_u, cmap=cm.coolwarm,linewidth=0, antialiased=False, alpha=0.5)
            surf = ax.plot_surface(ms_x,ms_t, u_true, color="k",linewidth=0, antialiased=False, alpha=0.5)

            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

            fig.colorbar(surf, shrink=0.5, aspect=5)

            save_path = os.path.join(save_dir, "pred_{}".format(epoch))
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(save_path)