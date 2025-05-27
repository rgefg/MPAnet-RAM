import torch
import numpy as np

from torch.cuda import amp
from ignite.engine import Engine
from ignite.engine import Events
from torch.autograd import no_grad
from utils.calc_acc import calc_acc
from torch.nn import functional as F


def create_train_engine(model, optimizer, non_blocking=False):
    device = torch.device("cuda", torch.cuda.current_device())
    scaler = amp.GradScaler()  # 创建梯度缩放器，放在这里，每个Engine实例独立

    def _process_func(engine, batch):           #训练过程的定义
        model.train()

        data, labels, cam_ids, img_paths, img_ids = batch
        epoch = engine.state.epoch

        data = data.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        cam_ids = cam_ids.to(device, non_blocking=non_blocking)

        optimizer.zero_grad()

        with amp.autocast():
            loss, metric = model(data, labels,
                                 cam_ids=cam_ids,
                                 epoch=epoch)

        # 使用scaler处理反向传播，混合精度
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        return metric          #指标字典（供日志记录使用）

    return Engine(_process_func)



def create_eval_engine(model, non_blocking=False):
    device = torch.device("cuda", torch.cuda.current_device())

    def _process_func(engine, batch):         #验证阶段，传入engine的处理函数有所不同
        model.eval()
        data, labels, cam_ids, img_paths = batch[:4]              #不要img_id，评估时不用区分同一个label的不同图片
        data = data.to(device, non_blocking=non_blocking)
        with no_grad():
            feat = model(data, labels,cam_ids=cam_ids.to(device, non_blocking=non_blocking))
        return feat.detach().float().cpu(), labels, cam_ids, np.array(img_paths)
    engine=Engine(_process_func)
    @engine.on(Events.EPOCH_STARTED)              #clear_data传入on函数，这是评估开始时的操作，清除缓存，与训练每个epoch开始不同（注意区分不同engine与不同event的操作），只会监听自己生命周期（即trainer调用evalator后）的事件。
    def clear_data(engine):
        # feat list
        if not hasattr(engine.state, "feat_list"):
            setattr(engine.state, "feat_list", [])         #epoch开始时（只有一个epoch，也就是评估开始）创建feat_list,如果已有，删去旧的特征集合（（batchsize*batchnum）*featurresize）
        else:
            engine.state.feat_list.clear()

        # id_list
        if not hasattr(engine.state, "id_list"):
            setattr(engine.state, "id_list", [])
        else:
            engine.state.id_list.clear()

        # cam list
        if not hasattr(engine.state, "cam_list"):
            setattr(engine.state, "cam_list", [])
        else:
            engine.state.cam_list.clear()

        # img path list
        if not hasattr(engine.state, "img_path_list"):
            setattr(engine.state, "img_path_list", [])
        else:
            engine.state.img_path_list.clear()

    @engine.on(Events.ITERATION_COMPLETED)
    def store_data(engine):
        engine.state.feat_list.append(engine.state.output[0])
        engine.state.id_list.append(engine.state.output[1])
        engine.state.cam_list.append(engine.state.output[2])
        engine.state.img_path_list.append(engine.state.output[3])

    return engine
