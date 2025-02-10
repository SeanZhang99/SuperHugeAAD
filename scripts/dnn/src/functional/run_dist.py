        step = 0
        best_loss = math.inf
        best_test_accu = torch.zeros((1,), device=rank)
        epidx_best_test_accu = 1
        best_val_accu = torch.zeros((1,), device=rank)
        test_accu_at_best_val = torch.zeros((1,), device=rank)
        epidx_best_val_accu = 1

        early_stop_idx = 0
        for epidx in trange(numEpoch) if rank == 0 else range(numEpoch):
            dist.barrier()
            # train
            if train_sampler is not None:
                train_sampler.set_epoch(epidx)
            net.train()
            running_loss = []
            val_loss = []
            test_loss = []
            training_classwise_hit = dict(
                zip(
                    range(numCategories),
                    [
                        0.0,
                    ]
                    * numCategories,
                )
            )
            val_classwise_hit = dict(
                zip(
                    range(numCategories),
                    [
                        0.0,
                    ]
                    * numCategories,
                )
            )
            test_classwise_hit = dict(
                zip(
                    range(numCategories),
                    [
                        0.0,
                    ]
                    * numCategories,
                )
            )

            trainig_classwise_count = dict(
                zip(
                    range(numCategories),
                    [
                        0.0,
                    ]
                    * numCategories,
                )
            )
            val_classwise_count = dict(
                zip(
                    range(numCategories),
                    [
                        0.0,
                    ]
                    * numCategories,
                )
            )
            test_classwise_count = dict(
                zip(
                    range(numCategories),
                    [
                        0.0,
                    ]
                    * numCategories,
                )
            )

            i = 0
            y_true = torch.empty(0, dtype=torch.int64, device=rank)
            y_pred = torch.empty(0, dtype=torch.int64, device=rank)
            dist.barrier()
            for s, trial, eeg, label, aux in traindataloader:
                i += 1
                optimizer.optimizer.zero_grad()
                eeg = eeg.to(rank)
                aux = aux.to(rank)
                label = label.to(rank)
                output = net(eeg, aux)
                loss = lossfn(output, label)
                loss.backward()
                optimizer.step()
                for name, param in net.named_parameters():
                    if param.grad is None:
                        print(name)
                step += 1
                running_loss.append(loss.detach().item())

                classify = torch.max(output, dim=1)[1]
                accuracy = (classify == label).float()
                y_true = torch.cat((y_true, label), dim=0)
                y_pred = torch.cat((y_pred, classify), dim=0)
                for pred, true in zip(classify, [int(x) for x in label]):
                    trainig_classwise_count[true] += 1
                    training_classwise_hit[true] += int(pred == true)
                dist.barrier()
            running_accu = 0.0
            for key in trainig_classwise_count.keys():
                if trainig_classwise_count[key] != 0:
                    running_accu += training_classwise_hit[key] / (
                        trainig_classwise_count[key] + 1e-8
                    )
            running_accu /= numCategories
            y_true = y_true.cpu()
            y_pred = y_pred.cpu()
            conf_mtx = confusion_matrix(y_true=y_true, y_pred=y_pred)
            if rank == 0:
                print("\nTraining Confusion matrix:")
                print(className)
                print(conf_mtx)

            # validation
            dist.barrier()
            net.eval()
            with torch.no_grad():
                j = 0
                for s, trial, eeg, label, aux in valdataloader:
                    j += 1
                    eeg = eeg.to(rank)
                    aux = aux.to(rank)
                    label = label.to(rank)
                    output = net(eeg, aux)
                    loss = lossfn(output, label)
                    val_loss.append(loss.item())
                    classify = torch.max(output, dim=1)[1]
                    accuracy = (classify == label).float()
                    for pred, true in zip(classify, [int(x) for x in label]):
                        val_classwise_count[true] += 1
                        val_classwise_hit[true] += int(pred == true)
                    dist.barrier()
                val_accu = 0.0
                for key in trainig_classwise_count.keys():
                    if val_classwise_count[key] != 0:
                        val_accu += val_classwise_hit[key] / (
                            val_classwise_count[key] + 1e-8
                        )
                val_accu /= numCategories

                dist.barrier()
                # test
                y_true = torch.empty(0, dtype=torch.int64, device=rank)
                y_pred = torch.empty(0, dtype=torch.int64, device=rank)
                trialwise_accuracy = torch.zeros(
                    max(config["dataset"]["LOOsubjectList"]),
                    32,
                    dtype=torch.float,
                    device=rank,
                )
                k = 0
                test_sample_count = torch.zeros_like(trialwise_accuracy)
                for s, trial, eeg, label, aux in testdataloader:
                    k += 1
                    eeg = eeg.to(rank)
                    aux = aux.to(rank)
                    label = label.to(rank)
                    output = net(eeg, aux)
                    loss = lossfn(output, label)
                    test_loss.append(loss.item())

                    classify = torch.max(output, dim=1)[1]
                    accuracy = (classify == label).float()
                    y_true = torch.cat((y_true, label), dim=0)
                    y_pred = torch.cat((y_pred, classify), dim=0)
                    for batch_index in range(len(s)):
                        trialwise_accuracy[
                            s[batch_index], trial[batch_index] - 1
                        ] += accuracy[batch_index]
                        test_sample_count[s[batch_index], trial[batch_index] - 1] += 1
                    for pred, true in zip(classify, [int(x) for x in label]):
                        test_classwise_count[true] += 1
                        test_classwise_hit[true] += int(pred == true)
                    dist.barrier()
                test_accu = 0.0
                for key in trainig_classwise_count.keys():
                    if test_classwise_count[key] != 0:
                        test_accu += test_classwise_hit[key] / (
                            test_classwise_count[key] + 1e-8
                        )
                test_accu /= numCategories

                if rank == 0:
                    y_true = y_true.contiguous()
                    y_pred = y_pred.contiguous()
                    y_true = y_true.cpu()
                    y_pred = y_pred.cpu()
                    conf_mtx = confusion_matrix(y_true=y_true, y_pred=y_pred)
                    if test_accu > best_test_accu:
                        best_test_accu = test_accu
                        epidx_best_test_accu = epidx
                        best_trialwise_accuracy: Tensor = trialwise_accuracy
                        best_test_sample_count: Tensor = test_sample_count
                        best_conf_matrix = conf_mtx
                    if val_accu > best_val_accu:
                        best_val_accu = val_accu
                        test_accu_at_best_val = test_accu
                        epidx_best_val_accu = epidx
                    print(
                        f"""
    |epoch {epidx+1:03d}|testing on subject {test_subject[0]:02d}|fold {fold:02d}|
    training accuracy {running_accu:.3f}|validation accuracy {val_accu:.3f}|test accuracy {test_accu:.3f}|
    best val accuracy {best_val_accu:.3f} in epoch {epidx_best_val_accu+1:03d}|
    best test accuracy {best_test_accu:.3f} in epoch {epidx_best_test_accu+1:03d}|
    current learning rate {optimizer.optimizer.param_groups[0]['lr']:.5e}
            """
                    )

                    trainAccu[0, epidx % 10] = running_accu
                    testAccu[0, epidx % 10] = test_accu
                    testAccu[0, -3] = test_accu_at_best_val
                    testAccu[0, -2] = best_test_accu
                    testAccu[0, -1] = torch.mean(testAccu[0, :10])
                    valAccu[0, epidx % 10] = val_accu
                    valAccu[0, -2] = best_val_accu
                    valAccu[0, -1] = torch.mean(valAccu[0, :10])
            dist.barrier()
            optimizer.step(epoch=epidx)
            gc.collect()

        if rank == 0:
            for tensor_, name_, mode_ in (
                (testAccu, "subjectwise", "test"),
                (trainAccu, "subjectwise", "train"),
                (valAccu, "subjectwise", "val"),
                (best_trialwise_accuracy, "trialwise", "test"),
                (best_test_sample_count, "sample_count", "test"),
                (best_conf_matrix, "conf_mtx", "test"),
            ):
                if type(tensor_) is torch.Tensor:
                    tensor_ = tensor_.cpu().numpy()
                df = pd.DataFrame(tensor_)  # type: ignore
                df.to_excel(
                    os.path.join(
                        excel_filepath,
                        f"winlen_{winlen:02.1f}_{name_:s}_{test_subject[0]:02d}_fold_{fold:d}_{mode_:s}.xlsx",
                    )
                )

