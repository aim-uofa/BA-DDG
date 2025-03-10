import torch


def recursive_to(obj, device):
    if isinstance(obj, torch.Tensor):
        try:
            return obj.cuda(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [recursive_to(o, device=device) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(o, device=device) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, device=device) for k, v in obj.items()}

    else:
        return obj
    

class CrossValidation(object):

    def __init__(self, config, num_cvfolds, model_factory):
        super().__init__()
        self.num_cvfolds = num_cvfolds
        self.models = [
            model_factory(config)
            for _ in range(num_cvfolds)
        ]
        self.optimizers = []
        self.schedulers = []
        
        for model in self.models:
                optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.999))
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=10, min_lr=1e-6)
                self.optimizers.append(optimizer)
                self.schedulers.append(scheduler)

    def get(self, fold):
        return self.models[fold], self.optimizers[fold], self.schedulers[fold]

    def to(self, device):
        for m in self.models:
            m.to(device)
        return self

    def state_dict(self):
        return {
            'models': [m.state_dict() for m in self.models],
            'optimizers': [o.state_dict() for o in self.optimizers],
            'schedulers': [s.state_dict() for s in self.schedulers],
        }

    def load_mpnn_state_dict(self, state_dict):
        for obj in self.models:
            obj.mpnn.load_state_dict(state_dict['model_state_dict'], strict=False)

    def load_state_dict_inference(self, state_dict):
        for sd, obj in zip(state_dict['models'], self.models):
            obj.load_state_dict(sd, strict=False)

    def load_state_dict(self, state_dict):
        for sd, obj in zip(state_dict['models'], self.models):
            obj.load_state_dict(sd, strict=False)
        for sd, obj in zip(state_dict['optimizers'], self.optimizers):
            obj.load_state_dict(sd)
        for sd, obj in zip(state_dict['schedulers'], self.schedulers):
            obj.load_state_dict(sd)