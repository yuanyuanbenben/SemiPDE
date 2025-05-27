import matplotlib.pyplot as plt
import numpy as np
import torch
import os

mode = 'rcd0.05'

incom_loss = 0
non_loss = 0
para_pic = 0
mse = 1


if mode == 'gfn':
    data_incom = np.load("./outputs/gfn_incom_64_16_50_0.100_0.001_0.005.npy")
    # data_non = np.load("./outputs/rcd_non_2.npy")
    # data_l2 = np.load("./outputs/rcd_norm.npy")
    # data_ortho = np.load("./outputs/rcd_ortho.npy")
    data_non = np.load("outputs/gfn_64_16_50_0.100_0.001_0.005.npy")
    leng = data_incom.shape[1] - 10
    epoch_list = np.linspace(0,leng*5,10,endpoint=False) + 5
    if not os.path.isdir('pic'):
        os.mkdir('pic')



    if incom_loss:
        plt.plot(epoch_list,data_incom[0,:leng],epoch_list,data_incom[1,:leng])
        plt.legend(labels = ['train loss','test loss'],loc=1)
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.savefig("./pic/gfn_incom_loss_64_16_50_0.100_0.001_0.005.png")
        plt.show()

    elif non_loss:
        plt.plot(epoch_list,data_non[0,:leng],epoch_list,data_non[1,:leng])
        plt.legend(labels = ['train loss','test loss'],loc=1)
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.savefig("./pic/gfn_loss_64_16_50_0.100_0.001_0.005.png")
        plt.show()

    elif para_pic:
        plt.plot(epoch_list,data_incom[2,:leng],epoch_list,data_incom[3,:leng],epoch_list,
        data_non[2,:leng],epoch_list,data_non[3,:leng],epoch_list,np.ones(leng)*0.001,'k--',epoch_list,np.ones(leng)*0.005,'k--')
        plt.legend(labels = ['incom D1','incom D2','semiPDE D1','semiPDE D2'],loc=(0.72,0.05))
        #plt.ylim(0,2.5)
        plt.ylabel('Parameter value')
        plt.xlabel('Epoch')
        plt.savefig("./pic/gfn_64_16_50_0.100_0.001_0.005.png")

    if mse:
        checkpoint_non = torch.load('./checkpoint/ckpt_gfn_64_16_50_0.100_0.001_0.005.pth')
        param_non_D1 = checkpoint_non['D_1']
        param_non_D2 = checkpoint_non['D_2']

        checkpoint_incom = torch.load('./checkpoint/ckpt_gfn_incom_64_16_50_0.100_0.001_0.005.pth')
        param_incom_D1 = checkpoint_incom['D_1']
        param_incom_D2 = checkpoint_incom['D_2']

        print(abs(param_non_D1/1e-3 - 1))
        print(abs(param_non_D2/5e-3 - 1))
        print(abs(param_incom_D1/1e-3 - 1))
        print(abs(param_incom_D2/5e-3 - 1))
        
        

if mode == 'np':
    data_incom = np.load("./outputs/np_incom_128_32_25_0.100_0.007_0.005.npy")
    # data_non = np.load("./outputs/rcd_non_2.npy")
    # data_l2 = np.load("./outputs/rcd_norm.npy")
    # data_ortho = np.load("./outputs/rcd_ortho.npy")
    data_non = np.load("outputs/np_128_32_25_0.100_0.007_0.005.npy")
    leng = data_incom.shape[1]
    epoch_list = np.linspace(0,leng*5,20,endpoint=False) + 5
    if not os.path.isdir('pic'):
        os.mkdir('pic')

    if incom_loss:
        plt.plot(epoch_list,data_incom[0,:leng],epoch_list,data_incom[1,:leng])
        plt.legend(labels = ['train loss','test loss'],loc=1)
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.savefig("./pic/np_incom_loss_128_32_25_0.100_0.007_0.005.png")
        plt.show()

    elif non_loss:
        plt.plot(epoch_list,data_non[0,:leng],epoch_list,data_non[1,:leng])
        plt.legend(labels = ['train loss','test loss'],loc=1)
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.savefig("./pic/np_loss_128_32_25_0.100_0.007_0.005.png")
        plt.show()

    elif para_pic:
        plt.plot(epoch_list,data_incom[2,:leng],epoch_list,data_incom[3,:leng],epoch_list,
        data_non[2,:leng],epoch_list,data_non[3,:leng],epoch_list,np.ones(leng)*0.007,'k--',epoch_list,np.ones(leng)*0.00533,'k--')
        plt.legend(labels = ['incom D1','incom D2','semiPDE D1','semiPDE D2'],loc=(0.72,0.5))
        #plt.ylim(0,2.5)
        plt.ylabel('Parameter value')
        plt.xlabel('Epoch')
        plt.savefig("./pic/np_128_32_25_0.100_0.007_0.005.png")

    if mse:
        checkpoint_non = torch.load('./checkpoint/ckpt_np_128_32_25_0.100_0.007_0.005.pth')
        param_non_D1 = checkpoint_non['D_1']
        param_non_D2 = checkpoint_non['D_2']

        checkpoint_incom = torch.load('./checkpoint/ckpt_np_incom_128_32_25_0.100_0.007_0.005.pth')
        param_incom_D1 = checkpoint_incom['D_1']
        param_incom_D2 = checkpoint_incom['D_2']

        print(abs(param_non_D1/7e-3 - 1))
        print(abs(param_non_D2/5.33e-3 - 1))
        print(abs(param_incom_D1/7e-3 - 1))
        print(abs(param_incom_D2/5.33e-3 - 1))
        

if mode == 'rcd0.02':
    data_incom = np.load("./outputs/rcd_incom_1_1_20_0.020_1.000_0.100.npy")
    # data_non = np.load("./outputs/rcd_non_2.npy")
    # data_l2 = np.load("./outputs/rcd_norm.npy")
    # data_ortho = np.load("./outputs/rcd_ortho.npy")
    data_non = np.load("outputs/rcd_1_1_20_0.020_1.000_0.100.npy")
    data_nopena = np.load("outputs/rcd_nopena_1_1_20_0.020_1.000_0.100.npy")
    leng = data_incom.shape[1]
    epoch_list = np.linspace(0,leng*5,40,endpoint=False) + 5
    if not os.path.isdir('pic'):
        os.mkdir('pic')

    if incom_loss:
        plt.plot(epoch_list,data_incom[0,:leng],epoch_list,data_incom[1,:leng])
        plt.legend(labels = ['train loss','test loss'],loc=1)
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.savefig("./pic/rcd_incom_loss_1_1_20_0.020_1.000_0.100.png")
        plt.show()

    elif non_loss:
        plt.plot(epoch_list,data_non[0,:leng],epoch_list,data_non[1,:leng])
        plt.legend(labels = ['train loss','test loss'],loc=1)
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.savefig("./pic/rcd_loss_1_1_20_0.020_1.000_0.100.png")
        plt.show()

    elif para_pic:
        plt.plot(epoch_list,data_incom[2,:leng],epoch_list,data_incom[3,:leng],epoch_list,data_incom[4,:leng],epoch_list,
        data_non[2,:leng],epoch_list,data_non[3,:leng],epoch_list,data_non[4,:leng],epoch_list,np.ones(leng),'k--',epoch_list,np.ones(leng)*0.1,'k--',epoch_list,np.ones(leng)*0.1,'k--')
        plt.legend(labels = ['incom D1','incom D2','incom D3','semiPDE D1','semiPDE D2','semiPDE D3'],loc=(0.72,0.5))
        #plt.ylim(0,2.5)
        plt.ylabel('Parameter value')
        plt.xlabel('Epoch')
        plt.savefig("./pic/rcd_1_1_20_0.020_1.000_0.100.png")

    if mse:
        checkpoint_non = torch.load('./checkpoint/ckpt_rcd_1_1_20_0.020_1.000_0.100.pth')
        param_non_D1 = checkpoint_non['D_1']
        param_non_D2 = checkpoint_non['D_2']
        param_non_D3 = checkpoint_non['D_3']

        checkpoint_incom = torch.load('./checkpoint/ckpt_rcd_incom_1_1_20_0.020_1.000_0.100.pth')
        param_incom_D1 = checkpoint_incom['D_1']
        param_incom_D2 = checkpoint_incom['D_2']
        param_incom_D3 = checkpoint_incom['D_3']
        
        checkpoint_nopena = torch.load('./checkpoint/ckpt_rcd_nopena_1_1_20_0.020_1.000_0.100.pth')
        param_nopena_D1 = checkpoint_nopena['D_1']
        param_nopena_D2 = checkpoint_nopena['D_2']
        param_nopena_D3 = checkpoint_nopena['D_3']
        
        checkpoint_pinn = torch.load('./checkpoint/ckpt_rcd_PINN_para_1_1_20_0.020_1.000_0.100.pth')
        param_pinn_D1 = checkpoint_pinn['D_1']
        param_pinn_D2 = checkpoint_pinn['D_2']
        param_pinn_D3 = checkpoint_pinn['D_3']
        
        checkpoint_pinn_non = torch.load('./checkpoint/ckpt_rcd_PINN_nonpara_1_1_20_0.020_1.000_0.100.pth')
        param_pinn_D1_non = checkpoint_pinn_non['D_1']
        param_pinn_D2_non = checkpoint_pinn_non['D_2']
        param_pinn_D3_non = checkpoint_pinn_non['D_3']

        print(abs(param_non_D1-1)*1000)
        print(abs(param_non_D2-0.1)*1000)
        print(abs(param_non_D3-0.1)*1000)
        print(abs(param_incom_D1-1)*1000)
        print(abs(param_incom_D2-0.1)*1000)
        print(abs(param_incom_D3-0.1)*1000)
        print(abs(param_nopena_D1-1)*1000)
        print(abs(param_nopena_D2-0.1)*1000)
        print(abs(param_nopena_D3-0.1)*1000)
        print(abs(param_pinn_D1*10-1)*1000)
        print(abs(param_pinn_D2-0.1)*1000)
        print(abs(param_pinn_D3-0.1)*1000)
        print(abs(param_pinn_D1_non*10-1)*1000)
        print(abs(param_pinn_D2_non-0.1)*1000)
        print(abs(param_pinn_D3_non-0.1)*1000)
        
        
        
if mode == 'rcd0.05':
    data_incom = np.load("./outputs/rcd_incom_1_1_20_0.050_1.000_0.100.npy")
    # data_non = np.load("./outputs/rcd_non_2.npy")
    # data_l2 = np.load("./outputs/rcd_norm.npy")
    # data_ortho = np.load("./outputs/rcd_ortho.npy")
    data_non = np.load("outputs/rcd_1_1_20_0.050_1.000_0.100.npy")
    data_nopena = np.load("outputs/rcd_nopena_1_1_20_0.050_1.000_0.100.npy")
    leng = data_incom.shape[1]
    epoch_list = np.linspace(0,leng*5,40,endpoint=False) + 5
    if not os.path.isdir('pic'):
        os.mkdir('pic')

    if incom_loss:
        plt.plot(epoch_list,data_incom[0,:leng],epoch_list,data_incom[1,:leng])
        plt.legend(labels = ['train loss','test loss'],loc=1)
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.savefig("./pic/rcd_incom_loss_1_1_20_0.050_1.000_0.100.png")
        plt.show()

    elif non_loss:
        plt.plot(epoch_list,data_non[0,:leng],epoch_list,data_non[1,:leng])
        plt.legend(labels = ['train loss','test loss'],loc=1)
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.savefig("./pic/rcd_loss_1_1_20_0.050_1.000_0.100.png")
        plt.show()

    elif para_pic:
        plt.plot(epoch_list,data_incom[2,:leng],epoch_list,data_incom[3,:leng],epoch_list,data_incom[4,:leng],epoch_list,
        data_non[2,:leng],epoch_list,data_non[3,:leng],epoch_list,data_non[4,:leng],epoch_list,np.ones(leng),'k--',epoch_list,np.ones(leng)*0.1,'k--',epoch_list,np.ones(leng)*0.1,'k--')
        plt.legend(labels = ['incom D1','incom D2','incom D3','semiPDE D1','semiPDE D2','semiPDE D3'],loc=(0.72,0.5))
        #plt.ylim(0,2.5)
        plt.ylabel('Parameter value')
        plt.xlabel('Epoch')
        plt.savefig("./pic/rcd_1_1_20_0.050_1.000_0.100.png")

    if mse:
        checkpoint_non = torch.load('./checkpoint/ckpt_rcd_1_1_20_0.050_1.000_0.100.pth')
        param_non_D1 = checkpoint_non['D_1']
        param_non_D2 = checkpoint_non['D_2']
        param_non_D3 = checkpoint_non['D_3']

        checkpoint_incom = torch.load('./checkpoint/ckpt_rcd_incom_1_1_20_0.050_1.000_0.100.pth')
        param_incom_D1 = checkpoint_incom['D_1']
        param_incom_D2 = checkpoint_incom['D_2']
        param_incom_D3 = checkpoint_incom['D_3']
        
        checkpoint_nopena = torch.load('./checkpoint/ckpt_rcd_nopena_1_1_20_0.050_1.000_0.100.pth')
        param_nopena_D1 = checkpoint_nopena['D_1']
        param_nopena_D2 = checkpoint_nopena['D_2']
        param_nopena_D3 = checkpoint_nopena['D_3']
        
        checkpoint_pinn = torch.load('./checkpoint/ckpt_rcd_PINN_para_1_1_20_0.050_1.000_0.100.pth')
        param_pinn_D1 = checkpoint_pinn['D_1']
        param_pinn_D2 = checkpoint_pinn['D_2']
        param_pinn_D3 = checkpoint_pinn['D_3']
        
        checkpoint_pinn_non = torch.load('./checkpoint/ckpt_rcd_PINN_nonpara_1_1_20_0.050_1.000_0.100.pth')
        param_pinn_D1_non = checkpoint_pinn_non['D_1']
        param_pinn_D2_non = checkpoint_pinn_non['D_2']
        param_pinn_D3_non = checkpoint_pinn_non['D_3']
        
        print(abs(param_non_D1-1)*1000)
        print(abs(param_non_D2-0.1)*1000)
        print(abs(param_non_D3-0.1)*1000)
        print(abs(param_incom_D1-1)*1000)
        print(abs(param_incom_D2-0.1)*1000)
        print(abs(param_incom_D3-0.1)*1000)
        print(abs(param_nopena_D1-1)*1000)
        print(abs(param_nopena_D2-0.1)*1000)
        print(abs(param_nopena_D3-0.1)*1000)
        print(abs(param_pinn_D1*10-1)*1000)
        print(abs(param_pinn_D2-0.1)*1000)
        print(abs(param_pinn_D3-0.1)*1000)
        print(abs(param_pinn_D1_non*10-1)*1000)
        print(abs(param_pinn_D2_non-0.1)*1000)
        print(abs(param_pinn_D3_non-0.1)*1000)
        
        
if mode == 'rd':
    data_incom = np.load("./outputs/rd_incom_64_16_25_0.100_0.003.npy")
    # data_non = np.load("./outputs/rcd_non_2.npy")
    # data_l2 = np.load("./outputs/rcd_norm.npy")
    # data_ortho = np.load("./outputs/rcd_ortho.npy")
    data_non = np.load("outputs/rd_64_16_25_0.100_0.003.npy")
    leng = data_incom.shape[1]
    epoch_list = np.linspace(0,leng*5,20,endpoint=False) + 5
    if not os.path.isdir('pic'):
        os.mkdir('pic')

    if incom_loss:
        plt.plot(epoch_list,data_incom[0,:leng],epoch_list,data_incom[1,:leng])
        plt.legend(labels = ['train loss','test loss'],loc=1)
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.savefig("./pic/rd_incom_loss_64_16_25_0.100_0.003.png")
        plt.show()

    elif non_loss:
        plt.plot(epoch_list,data_non[0,:leng],epoch_list,data_non[1,:leng])
        plt.legend(labels = ['train loss','test loss'],loc=1)
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.savefig("./pic/rd_loss_64_16_25_0.100_0.003.png")
        plt.show()

    elif para_pic:
        plt.plot(epoch_list,data_incom[2,:leng],epoch_list,
        data_non[2,:leng],epoch_list,np.ones(leng)*0.003,'k--',)
        plt.legend(labels = ['incom D','semiPDE D'],loc=(0.72,0.5))
        #plt.ylim(0,2.5)
        plt.ylabel('Parameter value')
        plt.xlabel('Epoch')
        plt.savefig("./pic/rd_64_16_25_0.100_0.003.png")

    if mse:
        checkpoint_non = torch.load('./checkpoint/ckpt_rd_64_16_25_0.100_0.003.pth')
        param_non_D = checkpoint_non['D']
        

        checkpoint_incom = torch.load('./checkpoint/ckpt_rd_incom_64_16_25_0.100_0.003.pth')
        param_incom_D = checkpoint_incom['D']

        print(abs(param_non_D/0.003 - 1)) 
        print(abs(param_incom_D/0.003 - 1))
        
        

if mode == 'ns':
    data_incom = np.load("./outputs/ns_incom_1600_320_25_0.100.npy")
    # data_non = np.load("./outputs/rcd_non_2.npy")
    # data_l2 = np.load("./outputs/rcd_norm.npy")
    # data_ortho = np.load("./outputs/rcd_ortho.npy")
    data_non = np.load("outputs/ns_1600_320_25_0.100.npy")
    leng = data_incom.shape[1]
    epoch_list = np.linspace(0,leng*5,40,endpoint=False) + 5
    if not os.path.isdir('pic'):
        os.mkdir('pic')

    if incom_loss:
        plt.plot(epoch_list,data_incom[0,:leng],epoch_list,data_incom[1,:leng])
        plt.legend(labels = ['train loss','test loss'],loc=1)
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.savefig("./pic/ns_incom_loss_1600_320_25_0.100.png")
        plt.show()

    elif non_loss:
        plt.plot(epoch_list,data_non[0,:leng],epoch_list,data_non[1,:leng])
        plt.legend(labels = ['train loss','test loss'],loc=1)
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.savefig("./pic/ns_loss_1600_320_25_0.100.png")
        plt.show()

    elif para_pic:
        # plt.plot(epoch_list,data_incom[2,:leng],epoch_list,data_incom[3,:leng],
        #          epoch_list,data_incom[4,:leng],
        #          epoch_list,data_non[2,:leng],epoch_list,data_non[3,:leng],
        #          epoch_list,data_non[4,:leng],
        #          epoch_list,np.ones(leng)*0.006,'k--',epoch_list,np.ones(leng)*0.007,'k--',
        #          epoch_list,np.ones(leng)*0.008,'k--',)
        # plt.legend(labels = ['incom D1','incom D2','incom D3',
        #                      'semiPDE D1','semiPDE D2','semiPDE D3',],loc=(0.72,0.65))
        # #plt.ylim(0,2.5)
        # plt.ylabel('Parameter value')
        # plt.xlabel('Epoch')
        # plt.savefig("./pic/ns_1600_320_25_0.100.png")
        plt.plot(epoch_list,data_incom[5,:leng],epoch_list,data_incom[6,:leng],
                 epoch_list,data_incom[7,:leng],
                 epoch_list,data_non[5,:leng],epoch_list,data_non[6,:leng],
                 epoch_list,data_non[7,:leng],
                 epoch_list,np.ones(leng)*0.003,'k--',epoch_list,np.ones(leng)*0.004,'k--',
                 epoch_list,np.ones(leng)*0.005,'k--',)
        plt.legend(labels = ['incom D4','incom D5','incom D6',
                             'semiPDE D4','semiPDE D5','semiPDE D6',],loc=(0.72,0.65))
        #plt.ylim(0,2.5)
        plt.ylabel('Parameter value')
        plt.xlabel('Epoch')
        plt.savefig("./pic/ns_1600_320_25_0.100_1.png")

    if mse:
        checkpoint_non = torch.load('./checkpoint/ckpt_ns_1600_320_25_0.100.pth')
        param_non_D1 = checkpoint_non['D_1']
        param_non_D2 = checkpoint_non['D_2']

        checkpoint_incom = torch.load('./checkpoint/ckpt_ns_incom_1600_320_25_0.100.pth')
        param_incom_D1 = checkpoint_incom['D_1']
        param_incom_D2 = checkpoint_incom['D_2']

        print(abs(param_non_D1[0]/6e-3 - 1))
        print(abs(param_non_D1[1]/7e-3 - 1))
        print(abs(param_non_D1[2]/8e-3 - 1))
        print(abs(param_non_D2[0]/3e-3 - 1))
        print(abs(param_non_D2[1]/4e-3 - 1))
        print(abs(param_non_D2[2]/5e-3 - 1))
        print(abs(param_incom_D1[0]/6e-3 - 1))
        print(abs(param_incom_D1[1]/7e-3 - 1))
        print(abs(param_incom_D1[2]/8e-3 - 1))
        print(abs(param_incom_D2[0]/3e-3 - 1))
        print(abs(param_incom_D2[1]/4e-3 - 1))
        print(abs(param_incom_D2[2]/5e-3 - 1))