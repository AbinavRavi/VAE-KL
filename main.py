from dataloader import *
from model import *
from train import *

def model_run(patch_size, batch_size, odd_class, z, seed=123, log_var_std=0, n_epochs=5,
              model_h_size=(16, 32, 64, 256), exp_name="exp", folder_name="exp"):
    set_seed(seed)

    config = Config(
        patch_size=patch_size, batch_size=batch_size, odd_class=odd_class, z=z, seed=seed, log_var_std=log_var_std,
        n_epochs=n_epochs
    )

    device = torch.device("cuda")

    datasets_common_args = {
        "batch_size": batch_size,
        "target_size": patch_size,
        "input_slice": [1, ],
        "add_noise": True,
        "mask_type": "gaussian",  # 0.0, ## TODO
        "elastic_deform": False,
        "rnd_crop": True,
        "rotate": True,
        "color_augment": True,
        "add_slices": 0,
    }

    input_shape = (
        datasets_common_args["batch_size"], 1, datasets_common_args["target_size"], datasets_common_args["target_size"])

    train_set_args = {
        "base_dir": "../test_data/normal/",
        # "num_batches": 500,
        # "slice_offset": 20,
        "num_processes": 8,
    }
    test_set_normal_args = {
        "base_dir": "brats17/",
        # "num_batches": 100,
        "do_reshuffle": False,
        "mode": "val",
        "num_processes": 2,
        "slice_offset": 20,
        "label_slice": 2,
        "only_labeled_slices": False,
    }
    test_set_unormal_args = {
        "base_dir": "brats17/",
        # "num_batches": 100,
        "do_reshuffle": False,
        "mode": "val",
        "num_processes": 2,
        "slice_offset": 20,
        "label_slice": 2,
        "only_labeled_slices": True,
        "labeled_threshold": 10,
    }
    test_set_all_args = {
        "base_dir": "brats17_test/",
        # "num_batches": 50,
        "do_reshuffle": False,
        "mode": "val",
        "num_processes": 2,
        "slice_offset": 20,
        "label_slice": 2,
    }

    train_loader = BrainDataSet(**datasets_common_args, **train_set_args)
    test_loader_normal = BrainDataSet(**datasets_common_args, **test_set_normal_args)
    test_loader_abnorm = BrainDataSet(**datasets_common_args, **test_set_unormal_args)
    test_loader_all = BrainDataSet(**datasets_common_args, **test_set_all_args)

    model = VAE(input_size=input_shape[1:], h_size=model_h_size, z_dim=z).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = StepLR(optimizer, step_size=1)

    vlog = PytorchVisdomLogger(exp_name=exp_name)
    elog = PytorchExperimentLogger(base_dir=folder_name, exp_name=exp_name)

    elog.save_config(config, "config")

    for epoch in range(1, n_epochs + 1):
        train(epoch, model, optimizer, train_loader, device, vlog, elog, log_var_std)

    kl_roc, rec_roc, loss_roc, kl_pr, rec_pr, loss_pr, test_loss = test_slice(model, test_loader_normal,
                                                                              test_loader_abnorm, device,
                                                                              vlog, elog, input_shape, batch_size,
                                                                              log_var_std)

    with open(os.path.join(elog.result_dir, "results.json"), "w") as file_:
        json.dump({
            "kl_roc": kl_roc, "rec_roc": rec_roc, "loss_roc": loss_roc,
            "kl_pr": kl_pr, "rec_pr": rec_pr, "loss_pr": loss_pr,
        }, file_, indent=4)

    elog.save_model(model, "vae")

    test_pixel(model, test_loader_all, device, vlog, elog, input_shape, batch_size, log_var_std)

    print("All done....")


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    patch_size = 64
    batch_size = 64
    odd_class = 0
    z = 256
    seed = 123
    log_var_std = 0.

    model_run(patch_size, batch_size, odd_class, z, seed, log_var_std)

dataset_config = {
    "path" : './data/',
    "patchsize" : (64,64),
    "margin" : (80,80)
}

dataloader_config = {
    "batch_size": 8,
    "num_workers":1
}