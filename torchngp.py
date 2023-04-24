import torch
import argparse
import numpy as np 

from nerf.provider import NeRFDataset
from nerf.gui import NeRFGUI
from nerf.utils import *

from functools import partial
from loss import huber_loss

# from collections import namedtuple
# from recordclass import recordclass

#torch.autograd.set_detect_anomaly(True)


def torchngp(data_path,workspace,O_mode=True,test_mode=False,seed=0,iters=200,init_lr=1e-2,
             ckpt='latest',num_rays=4096,cuda_ray=True,max_steps=1024,num_steps=512,
             upsample_steps=0,update_extra_interval=16,max_ray_batch=4096,patch_size=1,
             fp16=False,ff=False,tcnn=False,color_space='srgb',preload=True,bound=2,
             scale=1,offset=[0, 0, 0],dt_gamma=1/128,min_near=0.2,density_thresh=10,
             bg_radius=-1,gui=False,W=128,H=128,radius=5,fovy=50,max_spp=64,error_map=True,
             clip_text='',rand_pose=-1):

    #Insert arg
    class dummyClass:

        def __init__(self,path, workspace, O,test,seed,iters,lr,ckpt,
                    num_rays,cuda_ray,max_steps,num_steps,upsample_steps,
                    update_extra_interval,max_ray_batch,patch_size,fp16,
                    ff,tcnn,color_space,preload,bound,scale,offset,
                    dt_gamma,min_near,density_thresh,bg_radius,gui,W,
                    H,radius,fovy,max_spp,error_map,clip_text,rand_pose):
            
            self.path = path
            self.workspace = workspace
            self.O = O
            self.test=test
            self.seed=seed 
            self.iters=iters
            self.lr = init_lr 
            self.ckpt = ckpt
            self.num_rays = num_rays
            self.cuda_ray = cuda_ray
            self.max_steps = max_steps
            self.num_steps = num_steps
            self.upsample_steps = upsample_steps
            self.update_extra_interval = update_extra_interval
            self.max_ray_batch = max_ray_batch
            self.patch_size = patch_size
            self.fp16 = fp16 
            self.ff = ff
            self.tcnn = tcnn 
            self.color_space = color_space
            self.preload = preload
            self.bound = bound
            self.scale = scale 
            self.offset = offset  
            self.dt_gamma = dt_gamma 
            self.min_near = min_near 
            self.density_thresh = density_thresh
            self.bg_radius = bg_radius 
            self.gui = gui 
            self.W = W
            self.H = H 
            self.radius = radius
            self.fovy = fovy 
            self.max_spp = max_spp
            self.error_map = error_map
            self.clip_text = clip_text
            self.rand_pose = rand_pose

    
    #Init
    opt = dummyClass(data_path,workspace,O_mode,test_mode,seed,iters,init_lr,ckpt,num_rays,
                      cuda_ray,max_steps,num_steps,upsample_steps,update_extra_interval,
                      max_ray_batch,patch_size,fp16,ff,tcnn,color_space,preload,bound,scale,
                      offset,dt_gamma,min_near,density_thresh,bg_radius,gui,W,H,radius,fovy,
                      max_spp,error_map,clip_text,rand_pose)

    # ManualInput = recordclass("ManualInput", 
    #                          ["path", "workspace", "O","test","seed","iters","lr","ckpt",
    #                           "num_rays","cuda_ray","max_steps","num_steps","upsample_steps",
    #                           "update_extra_interval","max_ray_batch","patch_size","fp16",
    #                           "ff","tcnn","color_space","preload","bound","scale","offset",
    #                           "dt_gamma","min_near","density_thresh","bg_radius","gui","W",
    #                           "H","radius","fovy","max_spp","error_map","clip_text","rand_pose"])
    
    # opt = ManualInput(data_path,workspace,O_mode,test_mode,seed,iters,init_lr,ckpt,num_rays,
    #                   cuda_ray,max_steps,num_steps,upsample_steps,update_extra_interval,
    #                   max_ray_batch,patch_size,fp16,ff,tcnn,color_space,preload,bound,scale,
    #                   offset,dt_gamma,min_near,density_thresh,bg_radius,gui,W,H,radius,fovy,
    #                   max_spp,error_map,clip_text,rand_pose)


    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True
    
    if opt.patch_size > 1:
        opt.error_map = False # do not use error_map if use patch-based training
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."


    if opt.ff:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --ff"
        from nerf.network_ff import NeRFNetwork
    elif opt.tcnn:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"
        from nerf.network_tcnn import NeRFNetwork
    else:
        from nerf.network import NeRFNetwork
    
    seed_everything(opt.seed)

    model = NeRFNetwork(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
    )
    
    print(model)
    print("==="*20)
    print(vars(opt))
    print("==="*20)


    criterion = torch.nn.MSELoss(reduction='none')
    #criterion = partial(huber_loss, reduction='none')
    #criterion = torch.nn.HuberLoss(reduction='none', beta=0.1) # only available after torch 1.10 ?

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if opt.test:
        
        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)

        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()
        
        else:
            test_loader = NeRFDataset(opt, device=device, type='test').dataloader()

            if test_loader.has_gt:
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
    
            trainer.test(test_loader, write_video=True) # test and save video
            
            trainer.save_mesh(resolution=256, threshold=10)
    
    else:

        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

        train_loader = NeRFDataset(opt, device=device, type='train').dataloader()

        # decay to 0.1 * init_lr at last iter step
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt, eval_interval=50)

        #This is being hotwired to go to transforms_train.json for GA purpose; hacky fix
        valid_loader = NeRFDataset(opt, device=device, type='val', downscale=1).dataloader()

        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        trainer.train(train_loader, valid_loader, max_epoch, skip_saving=False) #just reuse trainset for validation for now in GA

        # also test
        test_loader = NeRFDataset(opt, device=device, type='test').dataloader()
        
        if test_loader.has_gt:
            trainer.evaluate(test_loader) # blender has gt, so evaluate it.

        trainer.test(test_loader, write_video=True) # test and save video
        
        trainer.save_mesh(resolution=256, threshold=10)


        # if opt.gui:
        #     gui = NeRFGUI(opt, trainer, train_loader)
        #     gui.render()
        
        # else:
        # valid_loader = NeRFDataset(opt, device=device, type='val', downscale=1).dataloader()

        # max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        # trainer.train(train_loader, valid_loader, max_epoch)

        # # also test
        # test_loader = NeRFDataset(opt, device=device, type='test').dataloader()
        
        # if test_loader.has_gt:
        #     trainer.evaluate(test_loader) # blender has gt, so evaluate it.
        
        # trainer.test(test_loader, write_video=False, write_image=False) # test and save video
        
        # trainer.save_mesh(resolution=256, threshold=10)

    #Compute metrics
    PSNR_ = trainer.metrics[0]
    PSNR = PSNR_.V / PSNR_.N

    #Compute metrics
    LPIPS_ = trainer.metrics[1]
    LPIPS = LPIPS_.V / LPIPS_.N

    loss = np.sum(trainer.stats["loss"]) / len(trainer.stats["loss"])


    return PSNR,LPIPS,loss,trainer