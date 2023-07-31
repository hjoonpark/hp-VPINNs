###############################################################################
###############################################################################
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pyDOE import lhs
from GaussJacobiQuadRule_V3 import Jacobi, DJacobi, GaussLobattoJacobiWeights
import time
import os
from collections import OrderedDict

torch.manual_seed(1234)
device = torch.device("cuda:0")
###############################################################################
###############################################################################
class VPINN(nn.Module):
    def __init__(self, X_u_train, u_train, X_f_train, f_train, X_quad, W_quad, U_exact_total, F_exact_total,\
                 gridx, gridy, N_testfcn, X_test, u_test, layers):
        super().__init__()

        self.x = X_u_train[:,0:1].to(device)
        self.y = X_u_train[:,1:2].to(device)
        self.utrain = u_train.to(device)
        self.xquad  = X_quad[:,0:1].to(device)
        self.yquad  = X_quad[:,1:2].to(device)
        self.wquad  = W_quad.to(device)
        self.xf = X_f_train[:,0:1].to(device)
        self.yf = X_f_train[:,1:2].to(device)
        self.ftrain = f_train.to(device)
        self.xtest = X_test[:,0:1].to(device)
        self.ytest = X_test[:,1:2].to(device)
        self.utest = u_test
        self.N_testfcn = N_testfcn
        self.Nelementx = len(N_testfcn[0])
        self.Nelementy = len(N_testfcn[1])
        self.Ntestx = N_testfcn[0][0]
        self.Ntesty = N_testfcn[1][0]
        self.U_ext_total = U_exact_total
        self.F_ext_total = F_exact_total
        self.gridx = gridx.to(device)
        self.gridy = gridy.to(device)
       
        self.layers = layers
        self.net = list()
        for i in range(len(layers)-1):
            self.net.append(("layer_%d" % i, nn.Linear(layers[i], layers[i+1])))
            if i < len(layers)-2:
                self.net.append(("activation_%d" % i, nn.Tanh()))
        self.net = nn.Sequential(OrderedDict(self.net)).to(device)

        self.optimizer_adam = torch.optim.Adam(self.net.parameters(), lr=0.001)
        
###############################################################################
    def forward(self, x, y, x_f, y_f, x_test, y_test):
        self.u_pred_boundary = self.net_u(x, y)
        self.f_pred = self.net_f(x_f, y_f)
        self.u_test = self.net_u(x_test, y_test)

        self.varloss_total = 0
        gridx = self.gridx
        gridy = self.gridy
        for ex in range(self.Nelementx):
            for ey in range(self.Nelementy):
                F_ext_element  = self.F_ext_total[ex, ey]
                Ntest_elementx = self.N_testfcn[0][ex]
                Ntest_elementy = self.N_testfcn[1][ey]
                
                x_quad_element = gridx[ex] + (gridx[ex+1]-gridx[ex])/2*(self.xquad+1)
                y_quad_element = gridy[ey] + (gridy[ey+1]-gridy[ey])/2*(self.yquad+1)
                jacobian_x     = ((gridx[ex+1]-gridx[ex])/2)
                jacobian_y     = ((gridy[ey+1]-gridy[ey])/2)
                jacobian       = ((gridx[ex+1]-gridx[ex])/2)*((gridy[ey+1]-gridy[ey])/2)
                
                u_NN_quad_element = self.net_u(x_quad_element, y_quad_element)
                d1xu_NN_quad_element, d2xu_NN_quad_element = self.net_dxu(x_quad_element, y_quad_element)
                d1yu_NN_quad_element, d2yu_NN_quad_element = self.net_dyu(x_quad_element, y_quad_element)

                testx_quad_element = self.Test_fcnx(Ntest_elementx, self.xquad)
                d1testx_quad_element, d2testx_quad_element = self.dTest_fcn(Ntest_elementx, self.xquad)
                testy_quad_element = self.Test_fcny(Ntest_elementy, self.yquad)
                d1testy_quad_element, d2testy_quad_element = self.dTest_fcn(Ntest_elementy, self.yquad)
    
                integrand_1 = d2xu_NN_quad_element + d2yu_NN_quad_element
                
                if var_form == 0:
                    U_NN_element = torch.tensor([[jacobian*torch.sum(\
                                    self.wquad[:,0:1]*testx_quad_element[r]*self.wquad[:,1:2]*testy_quad_element[k]*integrand_1) \
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementy)])

                if var_form == 1:
                    U_NN_element_1 = torch.tensor([[jacobian/jacobian_x*torch.sum(\
                                    self.wquad[:,0:1]*d1testx_quad_element[r]*self.wquad[:,1:2]*testy_quad_element[k]*d1xu_NN_quad_element) \
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementy)])
                    U_NN_element_2 = torch.tensor([[jacobian/jacobian_y*torch.sum(\
                                    self.wquad[:,0:1]*testx_quad_element[r]*self.wquad[:,1:2]*d1testy_quad_element[k]*d1yu_NN_quad_element) \
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementy)])
                    U_NN_element = - U_NN_element_1 - U_NN_element_2
    
                if var_form == 2:
                    U_NN_element_1 = torch.tensor([[jacobian*torch.sum(\
                                    self.wquad[:,0:1]*d2testx_quad_element[r]*self.wquad[:,1:2]*testy_quad_element[k]*u_NN_quad_element) \
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementy)])
                    U_NN_element_2 = torch.tensor([[jacobian*torch.sum(\
                                    self.wquad[:,0:1]*testx_quad_element[r]*self.wquad[:,1:2]*d2testy_quad_element[k]*u_NN_quad_element) \
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementy)])
                    U_NN_element = U_NN_element_1 + U_NN_element_2

                Res_NN_element = (U_NN_element - F_ext_element).view(1,-1)
                loss_element = torch.mean(torch.square(Res_NN_element))
                self.varloss_total = self.varloss_total + loss_element
 
        self.lossb = torch.mean(torch.square(self.utrain - self.u_pred_boundary))
        self.lossv = self.varloss_total
        self.lossp = torch.mean(torch.square(self.f_pred - self.ftrain))
        
        if scheme == 'VPINNs':
            self.loss  = 10*self.lossb + self.lossv 
        if scheme == 'PINNs':
            self.loss  = 10*self.lossb + self.lossp 
        return self.loss

    def net_u(self, x, y):
        xy = torch.cat([x,y], dim=1)
        u = self.net(xy)
        return u

    def net_dxu(self, x, y):
        u   = self.net_u(x, y)
        d1xu = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        d2xu = torch.autograd.grad(d1xu, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        return d1xu, d2xu

    def net_dyu(self, x, y):
        u    = self.net_u(x, y)
        d1yu = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        d2yu = torch.autograd.grad(d1yu, y, grad_outputs=torch.ones_like(d1yu), retain_graph=True, create_graph=True)[0]
        return d1yu, d2yu

    def net_f(self, x, y):
        u    = self.net_u(x, y)
        d1xu = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        d2xu = torch.autograd.grad(d1xu, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        d1yu = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        d2yu = torch.autograd.grad(d1yu, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        ftemp = d2xu + d2yu
        return ftemp

    def Test_fcnx(self, N_test,x):
        test_total = []
        for n in range(1,N_test+1):
            test  = Jacobi(n+1,0,0,x) - Jacobi(n-1,0,0,x)
            test = test.to(device)
            test_total.append(test)
        test_total = torch.cat(test_total)
        return test_total

    def Test_fcny(self, N_test,y):
        test_total = []
        for n in range(1,N_test+1):
            test  = Jacobi(n+1,0,0,y) - Jacobi(n-1,0,0,y)
            test = test.to(device)
            test_total.append(test)
        test_total = torch.cat(test_total)
        return test_total

    def dTest_fcn(self, N_test,x):
        d1test_total = []
        d2test_total = []
        for n in range(1,N_test+1):
            if n==1:
                d1test = ((n+2)/2)*Jacobi(n,1,1,x)
                d2test = ((n+2)*(n+3)/(2*2))*Jacobi(n-1,2,2,x)
                d1test_total.append(d1test)
                d2test_total.append(d2test)
            elif n==2:
                d1test = ((n+2)/2)*Jacobi(n,1,1,x) - ((n)/2)*Jacobi(n-2,1,1,x)
                d2test = ((n+2)*(n+3)/(2*2))*Jacobi(n-1,2,2,x)
                d1test_total.append(d1test)
                d2test_total.append(d2test)    
            else:
                d1test = ((n+2)/2)*Jacobi(n,1,1,x) - ((n)/2)*Jacobi(n-2,1,1,x)
                d2test = ((n+2)*(n+3)/(2*2))*Jacobi(n-1,2,2,x) - ((n)*(n+1)/(2*2))*Jacobi(n-3,2,2,x)
                d1test_total.append(d1test)
                d2test_total.append(d2test)
        d1test_total = torch.cat(d1test_total).to(device)
        d2test_total = torch.cat(d2test_total).to(device)
        return d1test_total, d2test_total

###############################################################################
    def train(self, nIter):
        start_time = time.time()
        for it in range(nIter):
            self.optimizer_adam.zero_grad()

            loss = self.forward(self.x, self.y, self.xf, self.yf, self.xtest, self.ytest)
            loss.backward()
            self.optimizer_adam.step()

            loss_his.append(loss.item())
#            if it % 1 == 0:
#                loss_value = self.sess.run(self.loss, tf_dict)
#                u_pred     = self.sess.run(self.u_test, tf_dict)
#                u_pred_his.append(u_pred)
            if it % 100 == 0:
                elapsed = time.time() - start_time
                str_print = ''.join(['It: %d, Loss: %.3e, Time: %.2f'])
                print(str_print % (it, loss.item(), elapsed))
                start_time = time.time()

    def predict(self, x_test, y_test):
        u_pred = self.net_u(x_test, y_test)
        # u_pred = self.sess.run(self.u_test, {self.x_test: self.xtest, self.y_test: self.ytest})
        return u_pred


###############################################################################
# =============================================================================
#                               Main
# =============================================================================    
if __name__ == "__main__":     
    print(">>>> START")
    scheme = "VPINNs"
    Net_layer = [2] + [5] * 3 + [1]
    var_form  = 1
    N_el_x = 4
    N_el_y = 4
    N_test_x = N_el_x*[5]
    N_test_y = N_el_y*[5]
    N_quad = 10
    N_bound = 80
    N_residual = 100   
    
    ###########################################################################
    def Test_fcn_x(n, x):
        test = Jacobi(n+1, 0, 0, x) - Jacobi(n-1, 0, 0, x)
        return test
    def Test_fcn_y(n,y):
       test  = Jacobi(n+1,0,0,y) - Jacobi(n-1,0,0,y)
       return test
    ###########################################################################
    omegax = 2*torch.pi
    omegay = 2*torch.pi
    r1 = 10
    def u_ext(x, y):
        # exact solution
        utemp = (0.1*torch.sin(omegax*x) + torch.tanh(r1*x)) * torch.sin(omegay*(y))
        return utemp

    def f_ext(x,y):
        gtemp = (-0.1*(omegax**2)*torch.sin(omegax*x) - (2*r1**2)*(torch.tanh(r1*x))/((torch.cosh(r1*x))**2))*torch.sin(omegay*(y))\
                +(0.1*torch.sin(omegax*x) + torch.tanh(r1*x)) * (-omegay**2 * torch.sin(omegay*(y)) )
        return gtemp

    ###########################################################################
    # Boundary points
    x_up = 2*torch.from_numpy(lhs(1,N_bound)-1).float() + 1
    y_up = torch.ones(len(x_up), dtype=torch.float32)[:,None]
    b_up = u_ext(x_up, y_up)
    x_up_train = torch.hstack((x_up, y_up))
    u_up_train = b_up

    x_lo = 2*torch.from_numpy(lhs(1,N_bound)-1).float() + 1
    y_lo = -1*torch.ones(len(x_lo), dtype=torch.float32)[:,None]
    b_lo = u_ext(x_lo, y_lo)
    x_lo_train = torch.hstack((x_lo, y_lo))
    u_lo_train = b_lo

    y_ri = 2*torch.from_numpy(lhs(1,N_bound)-1).float() + 1
    x_ri = torch.ones(len(y_ri), dtype=torch.float32)[:,None]
    b_ri = u_ext(x_ri, y_ri)
    x_ri_train = torch.hstack((x_ri, y_ri))
    u_ri_train = b_ri    

    y_le = 2*torch.from_numpy(lhs(1,N_bound)-1).float() + 1
    x_le = -1*torch.ones(len(y_le), dtype=torch.float32)[:,None]
    b_le = u_ext(x_le, y_le)
    x_le_train = torch.hstack((x_le, y_le))
    u_le_train = b_le    

    X_u_train = torch.cat((x_up_train, x_lo_train, x_ri_train, x_le_train))
    u_train = torch.cat((u_up_train, u_lo_train, u_ri_train, u_le_train))

    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    ###########################################################################
    # Residual points for PINNs
    grid_pt = torch.from_numpy(lhs(2,N_residual))
    xf = 2*grid_pt[:,0]-1
    yf = 2*grid_pt[:,1]-1
    ff = torch.tensor([ f_ext(xf[j],yf[j]) for j in range(len(yf))], dtype=torch.float32)
    X_f_train = torch.hstack((xf[:,None],yf[:,None])).float()
    f_train = ff[:,None]

    ###########################################################################
    # Quadrature points
    [X_quad, WX_quad] = GaussLobattoJacobiWeights(N_quad, 0, 0)
    Y_quad, WY_quad   = (X_quad, WX_quad)
    xx, yy            = torch.meshgrid(X_quad,  Y_quad)
    wxx, wyy          = torch.meshgrid(WX_quad, WY_quad)
    XY_quad_train     = torch.hstack((xx.flatten()[:,None],  yy.flatten()[:,None])).float()
    WXY_quad_train    = torch.hstack((wxx.flatten()[:,None], wyy.flatten()[:,None])).float()

    ###########################################################################
    # Construction of RHS for VPINNs
    NE_x, NE_y = N_el_x, N_el_y
    [x_l, x_r] = [-1, 1]
    [y_l, y_u] = [-1, 1]
    delta_x    = (x_r - x_l)/NE_x
    delta_y    = (y_u - y_l)/NE_y
    grid_x     = torch.tensor([ x_l + i*delta_x for i in range(NE_x+1)], dtype=torch.float32)
    grid_y     = torch.tensor([ y_l + i*delta_y for i in range(NE_y+1)], dtype=torch.float32)

    do_plot = 1
    s = 5
    if do_plot:
        plt.figure()
        plt.scatter(x_up, y_up, s=s, label="upper {}".format(len(x_up)))
        plt.scatter(x_lo, y_lo, s=s, label="lower {}".format(len(x_lo)))
        plt.scatter(x_ri, y_ri, s=s, label="right {}".format(len(x_ri)))
        plt.scatter(x_le, y_le, s=s, label="left {}".format(len(x_le)))
        plt.scatter(xf, yf, s=s, label="residual {}".format(len(xf)))
        plt.scatter(xx.flatten(), yy.flatten(), s=s, label="quadrature {}".format(len(xx)))
        plt.scatter(wxx.flatten(), wyy.flatten(), s=s, label="quadrature weights {}".format(len(wxx)))
        plt.scatter(grid_x.flatten(), grid_y.flatten(), s=s, label="RHS {}".format(len(grid_x)))
        plt.grid()
        plt.gca().set_aspect("equal")
        plt.title("Data points")
        plt.legend(loc="upper right")
        plt.xlim(right=3)
        plt.savefig(os.path.join(out_dir, "data_points"), dpi=150)

#    N_testfcn_total = [(len(grid_x)-1)*[N_test_x], (len(grid_y)-1)*[N_test_y]]
    N_testfcn_total = [N_test_x, N_test_y]
    print("N_testfcn_total:", len(N_test_x), len(N_test_y))

    #+++++++++++++++++++
    x_quad  = XY_quad_train[:,0:1]
    y_quad  = XY_quad_train[:,1:2]
    w_quad  = WXY_quad_train
    U_ext_total = []
    F_ext_total = []
    print("NE_x:", NE_x, ", NE_y:", NE_y)
    print("N_testfcn_total:", N_testfcn_total)
    print("grid_x:", grid_x)
    print("grid_y:", grid_y)
    print("x_quad:", x_quad.shape)
    print("y_quad:", y_quad.shape)
    for ex in range(NE_x):
        for ey in range(NE_y):
            Ntest_elementx  = N_testfcn_total[0][ex]
            Ntest_elementy  = N_testfcn_total[1][ey]

            x_quad_element = grid_x[ex] + (grid_x[ex+1]-grid_x[ex])/2*(x_quad+1)
            y_quad_element = grid_y[ey] + (grid_y[ey+1]-grid_y[ey])/2*(y_quad+1)
            jacobian       = ((grid_x[ex+1]-grid_x[ex])/2)*((grid_y[ey+1]-grid_y[ey])/2)
            
            testx_quad_element = torch.cat([ Test_fcn_x(n,x_quad) for n in range(1, Ntest_elementx+1)])
            testy_quad_element = torch.cat([ Test_fcn_y(n,y_quad) for n in range(1, Ntest_elementy+1)])
    
            u_quad_element = u_ext(x_quad_element, y_quad_element)
            f_quad_element = f_ext(x_quad_element, y_quad_element)
            
            # U_ext_element = torch.asarray([[jacobian*torch.sum(w_quad[:,0:1]*testx_quad_element[r]*w_quad[:,1:2]*testy_quad_element[k]*u_quad_element) for r in range(Ntest_elementx)] for k in range(Ntest_elementy)])
            # F_ext_element0 = torch.asarray([[jacobian*torch.sum(w_quad[:,0:1]*testx_quad_element[r]*w_quad[:,1:2]*testy_quad_element[k]*f_quad_element) for r in range(Ntest_elementx)] for k in range(Ntest_elementy)])
            U_ext_element = torch.zeros((Ntest_elementx, Ntest_elementy), dtype=torch.float32)
            F_ext_element = torch.zeros((Ntest_elementx, Ntest_elementy), dtype=torch.float32)
            for k in range(Ntest_elementy):
                for r in range(Ntest_elementx):
                    u = jacobian * torch.sum(w_quad[:,0:1]*testx_quad_element[r]*w_quad[:,1:2]*testy_quad_element[k]*u_quad_element)
                    U_ext_element[k, r] = u

                    f = jacobian * torch.sum(w_quad[:,0:1]*testx_quad_element[r]*w_quad[:,1:2]*testy_quad_element[k]*f_quad_element)
                    F_ext_element[k, r] = f

            U_ext_total.append(U_ext_element)
            F_ext_total.append(F_ext_element)
    
#    U_ext_total = torch.reshape(U_ext_total, [NE_x, NE_y, N_test_y, N_test_x])
    F_ext_total = torch.cat(F_ext_total)
    F_ext_total = F_ext_total.view(NE_x, NE_y, N_test_y[0], N_test_x[0])
    print("F_ext_total:", F_ext_total.shape)
    ###########################################################################
    # Test points
    delta_test = 0.01
    xtest = torch.arange(x_l, x_r + delta_test, delta_test, dtype=torch.float32)
    ytest = torch.arange(y_l, y_u + delta_test, delta_test, dtype=torch.float32)
    data_temp = torch.tensor([[ [xtest[i],ytest[j],u_ext(xtest[i],ytest[j])] for i in range(len(xtest))] for j in range(len(ytest))], dtype=torch.float32)
    Xtest = data_temp.flatten()[0::3]
    Ytest = data_temp.flatten()[1::3]
    Exact = data_temp.flatten()[2::3]
    X_test = torch.hstack((Xtest[:,None],Ytest[:,None]))
    u_test = Exact[:,None]

    ###########################################################################
    # grid_x = torch.tensor(grid_x, requires_grad=True).float()
    # grid_y = torch.tensor(grid_x, requires_grad=True).float()

    X_u_train.requires_grad = True
    u_train.requires_grad = True
    X_f_train.requires_grad = True
    f_train.requires_grad = True
    XY_quad_train.requires_grad = True
    model = VPINN(X_u_train, u_train, X_f_train, f_train, XY_quad_train, WXY_quad_train,\
                  U_ext_total, F_ext_total, grid_x, grid_y, N_testfcn_total, X_test, u_test, Net_layer)
    
    u_pred_his, loss_his = [], []
    # model.train(1)
    model.train(10000 + 1)
    u_pred = model.predict(model.xtest, model.ytest)

    print("model:")
    print(model)
    print("u_pred:", u_pred.shape)
    ###########################################################################
    # =============================================================================
    #    Plotting
    # =============================================================================
    import numpy as np
    print("----- plotting begins -----")
    fontsize = 24
    fig = plt.figure(1)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
    plt.xlabel('$iteration$', fontsize = fontsize)
    plt.ylabel('$loss \,\, values$', fontsize = fontsize)
    plt.yscale('log')
    plt.grid(True)
    plt.plot(loss_his)
    plt.tick_params( labelsize = 20)
    #fig.tight_layout()
    fig.set_size_inches(w=11,h=11)
    plt.savefig(out_dir.join(['Poisson2D_',scheme,'_loss']))
    
    ###########################################################################
    X_u_train = X_u_train.detach().numpy()
    X_test = X_test.detach().numpy()
    u_test = u_test.detach().numpy()
    u_pred = u_pred.cpu().detach().numpy()

    x_train_plot, y_train_plot = zip(*X_u_train)
    print("x_train_plot:", X_u_train.min(), X_u_train.max())
    x_f_plot, y_f_plot = zip(*X_f_train)
    fig, ax = plt.subplots(1)
    if scheme == 'VPINNs':
        plt.scatter(x_train_plot, y_train_plot, color='red')
        for xc in grid_x:
            plt.axvline(x=xc, ymin=0.045, ymax=0.954, linewidth=1.5)
        for yc in grid_y:
            plt.axhline(y=yc, xmin=0.045, xmax=0.954, linewidth=1.5)
    if scheme == 'PINNs':
        plt.scatter(x_train_plot, y_train_plot, color='red')
        plt.scatter(x_f_plot,y_f_plot)
        plt.axhline(-1, linewidth=1, linestyle='--', color='k')
        plt.axhline(1, linewidth=1, linestyle='--', color='k')
        plt.axvline(-1, linewidth=1, linestyle='--', color='k')
        plt.axvline(1, linewidth=1, linestyle='--', color='k')
    plt.xlim([-1.1,1.1])
    plt.ylim([-1.1,1.1])
    plt.xlabel('$x$', fontsize = fontsize)
    plt.ylabel('$y$', fontsize = fontsize)
    #ax.set_aspect(1)
    ax.locator_params(nbins=5)
    plt.tick_params( labelsize = 20)
    #fig.tight_layout()
    fig.set_size_inches(w=11,h=11)
    plt.savefig(out_dir.join(['Poisson2D_',scheme,'_Domain']))
    
    ###########################################################################
    x_test_plot = np.asarray(np.split(X_test[:,0:1].flatten(),len(ytest)))
    y_test_plot = np.asarray(np.split(X_test[:,1:2].flatten(),len(ytest)))
    u_test_plot = np.asarray(np.split(u_test.flatten(),len(ytest)))
    u_pred_plot = np.asarray(np.split(u_pred.flatten(),len(ytest)))
    
    
    fontsize = 32
    labelsize = 26
    fig_ext, ax_ext = plt.subplots(constrained_layout=True)
    CS_ext = ax_ext.contourf(x_test_plot, y_test_plot, u_test_plot, 100, cmap='jet', origin='lower')
    cbar = fig_ext.colorbar(CS_ext, shrink=0.67)
    cbar.ax.tick_params(labelsize = labelsize)
    ax_ext.locator_params(nbins=8)
    ax_ext.set_xlabel('$x$' , fontsize = fontsize)
    ax_ext.set_ylabel('$y$' , fontsize = fontsize)
    plt.tick_params( labelsize = labelsize)
    ax_ext.set_aspect(1)
    #fig.tight_layout()
    fig_ext.set_size_inches(w=11,h=11)
    plt.savefig(out_dir.join(['Poisson2D_',scheme,'_Exact','.png']))
    
    
    
    fig_pred, ax_pred = plt.subplots(constrained_layout=True)
    CS_pred = ax_pred.contourf(x_test_plot, y_test_plot, u_pred_plot, 100, cmap='jet', origin='lower')
    cbar = fig_pred.colorbar(CS_pred, shrink=0.67)
    cbar.ax.tick_params(labelsize = labelsize)
    ax_pred.locator_params(nbins=8)
    ax_pred.set_xlabel('$x$' , fontsize = fontsize)
    ax_pred.set_ylabel('$y$' , fontsize = fontsize)
    plt.tick_params( labelsize = labelsize)
    ax_pred.set_aspect(1)
    #fig.tight_layout()
    fig_pred.set_size_inches(w=11,h=11)
    plt.savefig(out_dir.join(['Poisson2D_',scheme,'_Predict','.png']))
    
    
    
    fig_err, ax_err = plt.subplots(constrained_layout=True)
    CS_err = ax_err.contourf(x_test_plot, y_test_plot, abs(u_test_plot - u_pred_plot), 100, cmap='jet', origin='lower')
    cbar = fig_err.colorbar(CS_err, shrink=0.65, format="%.4f")
    cbar.ax.tick_params(labelsize = labelsize)
    ax_err.locator_params(nbins=8)
    ax_err.set_xlabel('$x$' , fontsize = fontsize)
    ax_err.set_ylabel('$y$' , fontsize = fontsize)
    plt.tick_params( labelsize = labelsize)
    ax_err.set_aspect(1)
    #fig.tight_layout()
    fig_err.set_size_inches(w=11,h=11)
    plt.savefig(out_dir.join(['Poisson2D_',scheme,'_PntErr','.png']))
    
    
    print(">>>> DONE")