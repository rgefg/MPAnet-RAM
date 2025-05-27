import logging
import os
import numpy as np
import torch
import scipy.io as sio

from ignite.engine import Events
from ignite.handlers import ModelCheckpoint
from ignite.handlers import Timer

from engine.engine import create_eval_engine
from engine.engine import create_train_engine
from engine.metric import AutoKVMetric
from utils.eval_sysu import eval_sysu
from utils.eval_regdb import eval_regdb
from utils.eval_CMG import eval_CMG
from configs.default.dataset import dataset_cfg
from configs.default.strategy import strategy_cfg

def get_trainer(dataset, model, optimizer, lr_scheduler=None, logger=None, writer=None, non_blocking=False, log_period=10,           #没有用tensorboard的writer来记录日志
                save_dir="checkpoints", prefix="model", gallery_loader=None, query_loader=None,
                eval_interval=None, start_eval=None, rerank=False):
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
    
    # trainer
    trainer=create_train_engine(model, optimizer, non_blocking)        #创建训练engine
    setattr(trainer, "rerank", rerank)

    # checkpoint handler,ignite的一个模型保存处理器
    handler=ModelCheckpoint(
    dirname=save_dir,
    filename_prefix=prefix,
    n_saved=5,    #最多保存几个
    create_dir=True,
    require_empty=False
)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {"model": model})

    # metric
    timer = Timer(average=True)
    kv_metric = AutoKVMetric()

    # evaluator
    evaluator = None
    if not type(eval_interval) == int:
        raise TypeError("The parameter 'validate_interval' must be type INT.")
    if not type(start_eval) == int:
        raise TypeError("The parameter 'start_eval' must be type INT.")
    if eval_interval > 0 and gallery_loader is not None and query_loader is not None:
        evaluator=create_eval_engine(model, non_blocking)                       #如果训练需要评估，开始训练时创建评估engine（只创建一次）

    @trainer.on(Events.STARTED)
    def train_start(engine):
        setattr(engine.state, "best_rank1", 0.0)    #设置best指标属性

    @trainer.on(Events.COMPLETED)         #训练结束时用eval engine进行一次完整评估测试
    def train_completed(engine):
        torch.cuda.empty_cache()

        # extract query feature
        evaluator.run(query_loader)     #engine自带的run，加载query

        q_feats = torch.cat(evaluator.state.feat_list, dim=0)
        q_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
        q_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
        q_img_paths = np.concatenate(evaluator.state.img_path_list, axis=0)

        # extract gallery feature
        evaluator.run(gallery_loader)
        g_feats = torch.cat(evaluator.state.feat_list, dim=0)
        g_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
        g_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
        g_img_paths = np.concatenate(evaluator.state.img_path_list, axis=0)
        #print("best rank1={:.2f}%".format(engine.state.best_rank1))

        if dataset == 'sysu':           #调用评估函数，feat是提取的样本特征向量
            perm = sio.loadmat(os.path.join(dataset_cfg.sysu.data_root, 'exp', 'rand_perm_cam.mat'))[
                'rand_perm_cam']                                                                          #只考虑跨视角匹配，即 Gallery 中同相机同ID的结果会被加上mask忽略掉
            eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=1, rerank=engine.rerank)
            eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=10, rerank=engine.rerank)
            eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=1, rerank=engine.rerank)
            eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=10, rerank=engine.rerank)
        elif dataset == 'regdb':
            print('infrared to visible')
            eval_regdb(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, rerank=engine.rerank)
            print('visible to infrared')
            eval_regdb(g_feats, g_ids, g_cams, q_feats, q_ids, q_cams, q_img_paths, rerank=engine.rerank)
        elif dataset == 'market':
            eval_regdb(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, rerank=engine.rerank)
        elif dataset == 'market':
            eval_regdb(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, rerank=engine.rerank)
        elif dataset == 'CMG':
            perm = np.load(os.path.join(dataset_cfg.CMG.data_root, 'npy', 'random_perm.npy'),allow_pickle=True)  # shape: [6, 1, N_ids, 10, num_shots]
            eval_CMG(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, num_shots=1, rerank=engine.rerank)
            eval_CMG(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, num_shots=10, rerank=engine.rerank)


        evaluator.state.feat_list.clear()
        evaluator.state.id_list.clear()
        evaluator.state.cam_list.clear()
        evaluator.state.img_path_list.clear()
        del q_feats, q_ids, q_cams, g_feats, g_ids, g_cams
        torch.cuda.empty_cache()

    @trainer.on(Events.EPOCH_STARTED)           #每个训练epoch开始时，清零metric与计时器
    def epoch_started_callback(engine):
        epoch = engine.state.epoch
        if model.mutual_learning:
            model.update_rate = min(100 / (epoch + 1), 1.0) * model.update_rate_

        kv_metric.reset()
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)                   #每个epoch结束，如果要评估，调用eval engine评估
    def epoch_completed_callback(engine):
        epoch = engine.state.epoch

        if lr_scheduler is not None:
            lr_scheduler.step()

        #if epoch % eval_interval == 0:
            #logger.info("Model saved at {}/{}_model_{}.pth".format(save_dir, prefix, epoch))

        if evaluator and epoch % eval_interval== 0 and epoch > start_eval:
            #只有满足 epoch % eval_interval == 0 时，才会真正用 eval_engine 提取特征并计算指标。其他时候特征不会被提取或评估，节省算力
            #evaluator 是在训练前定义好，但只在满足条件时才会调用 run()
            #每次 .run() 都会触发一次完整的事件链，监听器被执行。非 .run() 的 epoch，evaluator 虽然“在内存中存在”，但不活跃，不监听任何event
            torch.cuda.empty_cache()
            # extract query feature
            evaluator.run(query_loader)       #epoch完成时验证用的也是测试集

            q_feats = torch.cat(evaluator.state.feat_list, dim=0)                    #Query 特征张量（N × D），N 是图像数量，D 是特征维度
            q_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()                #Query 的 person ID，用于判断匹配是否正确
            q_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()               #Query 所属的相机 ID，用于过滤“同相机同 ID”的匹配（避免刷分）
            q_img_paths = np.concatenate(evaluator.state.img_path_list, axis=0)       #Query 图像路径（主要用于可视化或 debug）

            # extract gallery feature
            evaluator.run(gallery_loader)

            g_feats = torch.cat(evaluator.state.feat_list, dim=0)
            g_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
            g_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
            g_img_paths = np.concatenate(evaluator.state.img_path_list, axis=0)

            if dataset == 'sysu':    #sysumm01是单向的，regdb是双向的检索，记录map，r1，r5
                perm = sio.loadmat(os.path.join(dataset_cfg.sysu.data_root, 'exp', 'rand_perm_cam.mat'))[
                    'rand_perm_cam']
                mAP, r1, r5, _, _ = eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=1, rerank=engine.rerank)
                eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=10, rerank=engine.rerank)
                eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=1, rerank=engine.rerank)
                eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=10, rerank=engine.rerank)  
                #Re-ranking 通常是在计算完初步距离之后，再通过 k-reciprocal 最近邻方法做一次改进，提升 mAP
            elif dataset == 'regdb':
                print('infrared to visible')
                mAP, r1, r5, _, _ = eval_regdb(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, rerank=engine.rerank)
                print('visible to infrared')
                mAP, r1_, r5, _, _ = eval_regdb(g_feats, g_ids, g_cams, q_feats, q_ids, q_cams, q_img_paths, rerank=engine.rerank)
                r1 = (r1 + r1_) / 2
            elif dataset == 'market':
                mAP, r1, r5, _, _ = eval_regdb(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, rerank=engine.rerank)
            elif dataset=='CMG':
                perm = np.load(os.path.join(dataset_cfg.CMG.data_root, 'npy', 'random_perm.npy'),allow_pickle=True)
                mAP, r1, r5, _, _=eval_CMG(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm,num_shots=1, rerank=engine.rerank)
                eval_CMG(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm,num_shots=10, rerank=engine.rerank)

            
            if r1 > engine.state.best_rank1:
                engine.state.best_rank1 = r1
                print("now new best rk1 found!")
                print("best rank1={:.2f}%".format(engine.state.best_rank1))
                torch.save(model.state_dict(), "{}/model_best.pth".format(save_dir))                              #评估时保存最好的rk1

            #if writer is not None:
                #writer.add_scalar('eval/mAP', mAP, epoch)
                #writer.add_scalar('eval/r1', r1, epoch)
                #writer.add_scalar('eval/r5', r5, epoch)

            evaluator.state.feat_list.clear()
            evaluator.state.id_list.clear()
            evaluator.state.cam_list.clear()
            evaluator.state.img_path_list.clear()
            del q_feats, q_ids, q_cams, g_feats, g_ids, g_cams

            torch.cuda.empty_cache()

    @trainer.on(Events.ITERATION_COMPLETED)
    def iteration_complete_callback(engine):
        timer.step()

        # print(engine.state.output)
        kv_metric.update(engine.state.output)            #每次迭代后，更新loss，acc字典，output 应该就是当前batch的loss、acc打包成metric

        epoch = engine.state.epoch
        iteration = engine.state.iteration
        iter_in_epoch = iteration - (epoch - 1) * len(engine.state.dataloader)

        if iter_in_epoch % log_period == 0 and iter_in_epoch > 0:
            batch_size = engine.state.batch[0].size(0)
            speed = batch_size / timer.value()

            msg = "Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec" % (epoch, iter_in_epoch, speed)

            metric_dict = kv_metric.compute()

            # log output information
            if logger is not None:
                for k in sorted(metric_dict.keys()):
                    msg += "\t%s: %.4f" % (k, metric_dict[k])
                    if writer is not None:
                        writer.add_scalar('metric/{}'.format(k), metric_dict[k], iteration)      #每隔log_period 个batch打印一次日志。计算当前 batch 处理速度（samples/sec）
                #从 kv_metric 中拿出 loss/acc/其他指标。逐项打印到 msg 中，并写入 TensorBoard，最后用 logger.info 印整行日志
                logger.info(msg)
            kv_metric.reset()
            timer.reset()

    return trainer


