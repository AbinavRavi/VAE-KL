def test_slice(model, test_loader, test_loader_abnorm, device, vlog, elog, image_size, batch_size, log_var_std):
    model.eval()
    test_loss = []
    kl_loss = []
    rec_loss = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data["data"][0].float().to(device)
            recon_batch, mu, logstd = model(data)
            loss, kl, rec = loss_function(recon_batch, data, mu, logstd, log_var_std)
            test_loss += (kl + rec).tolist()
            kl_loss += kl.tolist()
            rec_loss += rec.tolist()
            # if i == 0:
            #     n = min(data.size(0), 8)
            #     comparison = torch.cat([data[:n],
            #                             recon_batch[:n]])
                # vlog.show_image_grid(comparison.cpu(),                                     name='reconstruction')
    test_loss_ab = []
    kl_loss_ab = []
    rec_loss_ab = []
    with torch.no_grad():
        for i, data in enumerate(test_loader_abnorm):
            data = data["data"][0].float().to(device)
            recon_batch, mu, logstd = model(data)
            loss, kl, rec = loss_function(recon_batch, data, mu, logstd, log_var_std)
            test_loss_ab += (kl + rec).tolist()
            kl_loss_ab += kl.tolist()
            rec_loss_ab += rec.tolist()
            # if i == 0:
            #     n = min(data.size(0), 8)
            #     comparison = torch.cat([data[:n],
            #                             recon_batch[:n]])
                # vlog.show_image_grid(comparison.cpu(),                                     name='reconstruction2')

    kl_roc, kl_pr = elog.get_classification_metrics(kl_loss + kl_loss_ab,
                                                    [0] * len(kl_loss) + [1] * len(kl_loss_ab),
                                                    )[0]
    rec_roc, rec_pr = elog.get_classification_metrics(rec_loss + rec_loss_ab,
                                                      [0] * len(rec_loss) + [1] * len(rec_loss_ab),
                                                      )[0]
    loss_roc, loss_pr = elog.get_classification_metrics(test_loss + test_loss_ab,
                                                        [0] * len(test_loss) + [1] * len(test_loss_ab),
                                                        )[0]

    return kl_roc, rec_roc, loss_roc, kl_pr, rec_pr, loss_pr, np.mean(test_loss)
