import copy
import logging
import numpy as np
import os
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from inc_net import ResNetCosineIncrementalNet,SimpleVitNet
from utils.toolkit import target2onehot, tensor2numpy, accuracy

num_workers = 8

class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self.class_increments=[]
        self._network = None

        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

    def eval_task(self):
        y_pred, y_true, task_pred, task_true = self._eval_cnn(self.test_loader)
        acc_total,grouped = self._evaluate(y_pred, y_true)
        task_acc = np.around(
            (task_pred == task_true).sum() * 100 / len(task_true), decimals=2
        ) if len(task_true) > 0 else 0
        per_task_acc = {}
        per_task_recall = {}
        for task_id in np.unique(task_true):
            pred_task_ids = task_pred[task_true == task_id]
            true_task_ids = task_true[task_true == task_id]
            per_task_acc[task_id] = np.around(
                (pred_task_ids == true_task_ids).sum() * 100 / len(true_task_ids), decimals=2
            )

            pred_task_ids = task_pred[task_pred == task_id]
            true_task_ids = task_true[task_pred == task_id]
            per_task_recall[task_id] = np.around(
                (pred_task_ids == true_task_ids).sum() * 100 / len(true_task_ids), decimals=2
            )

        return acc_total,grouped,y_pred[:,0],y_true,task_acc,per_task_acc,per_task_recall

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        task_pred, task_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs, targets)
            predicts = torch.topk(outputs["logits"], k=1, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

            task_pred.append(outputs['pred_task_ids'].cpu().numpy())
            task_true.append(outputs['real_task_ids'].cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true), np.concatenate(task_pred), np.concatenate(task_true)  
    
    def _evaluate(self, y_pred, y_true):
        ret = {}
        acc_total,grouped = accuracy(y_pred.T[0], y_true, self._known_classes,self.class_increments)
        return acc_total,grouped 
    
    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if args["model_name"]!='ncm':
            if args["model_name"]=='adapter' and '_adapter' not in args["convnet_type"]:
                raise NotImplementedError('Adapter requires Adapter backbone')
            if args["model_name"]=='ssf' and '_ssf' not in args["convnet_type"]:
                raise NotImplementedError('SSF requires SSF backbone')
            if args["model_name"]=='vpt' and '_vpt' not in args["convnet_type"]:
                raise NotImplementedError('VPT requires VPT backbone')

            if 'resnet' in args['convnet_type']:
                self._network = ResNetCosineIncrementalNet(args, True)
                self._batch_size=128
            else:
                self._network = SimpleVitNet(args, True)
                self._batch_size= args["batch_size"]
            
            self.weight_decay=args["weight_decay"] if args["weight_decay"] is not None else 0.0005
            self.min_lr=args['min_lr'] if args['min_lr'] is not None else 1e-8
        else:
            self._network = SimpleVitNet(args, True)
            self._batch_size= args["batch_size"]
        self.args=args

    def incremental_train(self, data_manager):
        self.total_classnum = data_manager.get_total_classnum()
        self._cur_task += 1
        self._classes_seen_so_far = self._known_classes + data_manager.get_task_size(self._cur_task)
        if self.args['use_RP']:
            #temporarily remove RP weights
            del self._network.fc
            self._network.fc=None
        self._network.update_fc(self._classes_seen_so_far) #creates a new head with a new number of classes (if CIL)
        self._network.update_map(self._cur_task, np.arange(self._known_classes, self._classes_seen_so_far))
        if self.is_dil == False:
            logging.info("Starting CIL Task {}".format(self._cur_task+1))
        logging.info("Learning on classes {}-{}".format(self._known_classes, self._classes_seen_so_far-1))
        self.class_increments.append([self._known_classes, self._classes_seen_so_far-1])
        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._classes_seen_so_far),source="train", mode="train", )
        self.train_loader = DataLoader(self.train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=num_workers)
        train_dataset_for_CPs = data_manager.get_dataset(np.arange(self._known_classes, self._classes_seen_so_far),source="train", mode="test", )
        self.train_loader_for_CPs = DataLoader(train_dataset_for_CPs, batch_size=self._batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._classes_seen_so_far), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=num_workers)
        if len(self._multiple_gpus) > 1:
            logging.info('Using multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_CPs)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_CPs):
        self._network.to(self._device)
        if self._cur_task == 0 and self.args["model_name"] in ['ncm','joint_linear']:
             self.freeze_backbone()
        if self.args["model_name"] in ['joint_linear','joint_full']: 
            #this branch updates using SGD on all tasks and should be using classes and does not use a RP head
            if self.args["model_name"] =='joint_linear':
                assert self.args['body_lr']==0.0
            self.show_num_params()
            optimizer = optim.SGD([{'params':self._network.convnet.parameters()},{'params':self._network.fc.parameters(),'lr':self.args['head_lr']}], 
                                        momentum=0.9, lr=self.args['body_lr'],weight_decay=self.weight_decay)
            scheduler=optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100000])
            logging.info("Starting joint training on all data using "+self.args["model_name"]+" method")
            self._init_train(train_loader, test_loader, optimizer, scheduler)
            self.show_num_params()
        else:
            #this branch is either CP updates only, or SGD on a PETL method first task only
            if self._cur_task == 0 and self.dil_init==False:
                if 'ssf' in self.args['convnet_type']:
                    self.freeze_backbone(is_first_session=True)
                if self.args["model_name"] != 'ncm':
                    #this will be a PETL method. Here, 'body_lr' means all parameters
                    self.show_num_params(verbose=True)
                    optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.args['body_lr'],weight_decay=self.weight_decay)
                    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
                    #train the PETL method for the first task:
                    logging.info("Starting PETL training on first task using "+self.args["model_name"]+" method")
                    self._init_train(train_loader, test_loader, optimizer, scheduler)
                    self.freeze_backbone()
                if self.args['use_RP'] and self.dil_init==False:
                    self.setup_RP()
            if self.is_dil and self.dil_init==False:
                self.dil_init=True
                self._network.fc.weight.data.fill_(0.0)
            self.replace_fc(train_loader_for_CPs)
            self.show_num_params(verbose=True)

    def freeze_backbone(self,is_first_session=False):
        # Freeze the parameters for ViT.
        if 'vit' in self.args['convnet_type']:
            if isinstance(self._network.convnet, nn.Module):
                for name, param in self._network.convnet.named_parameters():
                    if is_first_session:
                        if "head." not in name and "ssf_scale" not in name and "ssf_shift_" not in name: 
                            param.requires_grad = False
                    else:
                        param.requires_grad = False
        else:
            if isinstance(self._network.convnet, nn.Module):
                for name, param in self._network.convnet.named_parameters():
                    if is_first_session:
                        if "ssf_scale" not in name and "ssf_shift_" not in name: 
                            param.requires_grad = False
                    else:
                        param.requires_grad = False

    def show_num_params(self,verbose=False):
        # show total parameters and trainable parameters
        total_params = sum(p.numel() for p in self._network.parameters())
        logging.info(f'{total_params:,} total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} training parameters.')
        if total_params != total_trainable_params and verbose:
            for name, param in self._network.named_parameters():
                if param.requires_grad:
                    logging.info(f"{name}: {param.numel()}")

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)

    def setup_RP(self):
        self.initiated_G=False
        self._network.fc.use_RP=True
        if self.args['M']>0:
            #RP with M > 0
            M=self.args['M']
            self._network.fc.weight = nn.Parameter(torch.Tensor(self._network.fc.out_features, M).to(device='cuda')) #num classes in task x M
            self._network.fc.reset_parameters()
            if self.args.get('sparsity', 1) != 1:
                # bernoulli sparsity
                # self._network.fc.W_rand=torch.bernoulli(
                #     torch.full((self._network.fc.in_features, M), self.args['sparsity'])
                #     ).float().to(device='cuda')

                # column-wise sparsity
                num_active = int(self._network.fc.in_features * self.args['sparsity'])
                W_rand = torch.randn(self._network.fc.in_features, M).to(device='cuda')
                _, top_indices = torch.topk(W_rand, k=num_active, dim=0)
                mask = torch.zeros_like(W_rand, dtype=torch.bool)
                mask.scatter_(0, top_indices, True)
                W_rand = W_rand * mask

                # torch.nn.init.sparse_
                # W_rand = torch.randn(self._network.fc.in_features, M)
                # nn.init.sparse_(W_rand, sparsity=self.args['sparsity'], std=0.01)

                self._network.fc.W_rand = W_rand.to(device='cuda')
            elif self.args.get('rp_bottleneck', 0) != 0:
                # low rank RP
                bottleneck_size = int(self.args['rp_bottleneck'])
                W_rand_pre = torch.randn(self._network.fc.in_features, bottleneck_size)
                W_rand_post = torch.randn(bottleneck_size, M)
                self._network.fc.W_rand = (W_rand_pre @ W_rand_post).to(device='cuda')
            else:
                self._network.fc.W_rand=torch.randn(self._network.fc.in_features,M).to(device='cuda')
            self.W_rand=copy.deepcopy(self._network.fc.W_rand) #make a copy that gets passed each time the head is replaced
        else:
            #no RP, only decorrelation
            M=self._network.fc.in_features #this M is L in the paper
        self.Q=torch.zeros(M,self.total_classnum)
        self.G=torch.zeros(M,M)
    
    def replace_fc(self,trainloader):
        self._network = self._network.eval()

        if self.args['use_RP']:
            #these lines are needed because the CosineLinear head gets deleted between streams and replaced by one with more classes (for CIL)
            self._network.fc.use_RP=True
            if self.args['M']>0:
                self._network.fc.W_rand=self.W_rand
            else:
                self._network.fc.W_rand=None

        Features_f = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data,label)=batch
                data=data.cuda()
                label=label.cuda()
                embedding = self._network.convnet(data)
                Features_f.append(embedding.cpu())
                label_list.append(label.cpu())
        Features_f = torch.cat(Features_f, dim=0)
        label_list = torch.cat(label_list, dim=0)
        
        Y=target2onehot(label_list,self.total_classnum)
        if self.args['use_RP']:
            #print('Number of pre-trained feature dimensions = ',Features_f.shape[-1])
            if self.args['M']>0:
                Features_h=torch.nn.functional.relu(Features_f@ self._network.fc.W_rand.cpu())
            else:
                Features_h=Features_f
            
            # Save Features_h for PCA analysis
            # self._save_features_h(Features_h, label_list)
            
            cur_Q = Features_h.T @ Y
            cur_G = Features_h.T @ Features_h
            self.Q=self.Q+cur_Q
            self.G=self.G+cur_G
            ridge=self.optimise_ridge_parameter(Features_h,Y)
            Wo=torch.linalg.solve(self.G+ridge*torch.eye(self.G.size(dim=0)),self.Q).T #better nmerical stability than .inv
            self._network.fc.weight.data=Wo[0:self._network.fc.weight.shape[0],:].to(device='cuda')

            if self.args.get('use_task_wise_fc', False):
                cur_Wo = torch.linalg.solve(cur_G+ridge*torch.eye(cur_G.size(dim=0)),cur_Q).T
                self._network.init_task_wise_fc(self._cur_task, self.W_rand)
                self._network.task_wise_fc[self._cur_task].weight.data = \
                    cur_Wo[0:self._network.task_wise_fc[self._cur_task].weight.shape[0],:].to(device='cuda')
        else:
            # For non-RP methods, save Features_f as Features_h
            # self._save_features_h(Features_f, label_list)
            
            for class_index in np.unique(self.train_dataset.labels):
                data_index=(label_list==class_index).nonzero().squeeze(-1)
                if self.is_dil:
                    class_prototype=Features_f[data_index].sum(0)
                    self._network.fc.weight.data[class_index]+=class_prototype.to(device='cuda') #for dil, we update all classes in all tasks
                else:
                    #original cosine similarity approach of Zhou et al (2023)
                    class_prototype=Features_f[data_index].mean(0)
                    self._network.fc.weight.data[class_index]=class_prototype #for cil, only new classes get updated

    def _save_features_h(self, Features_h, label_list):
        """Save Features_h and corresponding labels for PCA analysis"""
        try:
            # Create log directory structure similar to trainer.py
            init_cls = 0 if self.args["init_cls"] == self.args["increment"] else self.args["init_cls"]
            log_dir = "logs/{}/{}/{}/{}/{}".format(
                self.args["model_name"], 
                self.args["dataset"], 
                init_cls, 
                self.args['increment'], 
                self.args['exp_name']
            )
            
            # Create features subdirectory
            features_dir = os.path.join(log_dir, "features_h")
            if not os.path.exists(features_dir):
                os.makedirs(features_dir)
            
            # Convert to numpy arrays
            features_h_np = Features_h.detach().cpu().numpy()
            labels_np = label_list.detach().cpu().numpy()
            
            # Save features and labels for this task
            features_filename = os.path.join(features_dir, f"task_{self._cur_task}_features_h.npy")
            labels_filename = os.path.join(features_dir, f"task_{self._cur_task}_labels.npy")
            
            np.save(features_filename, features_h_np)
            np.save(labels_filename, labels_np)
            
            logging.info(f"Saved Features_h for task {self._cur_task}: shape {features_h_np.shape}")
            logging.info(f"Saved to: {features_filename}")
            
        except Exception as e:
            logging.warning(f"Failed to save Features_h for task {self._cur_task}: {e}")

    def optimise_ridge_parameter(self,Features,Y):
        ridges=10.0**np.arange(-8,9)
        num_val_samples=int(Features.shape[0]*0.8)
        losses=[]
        Q_val=Features[0:num_val_samples,:].T @ Y[0:num_val_samples,:]
        G_val=Features[0:num_val_samples,:].T @ Features[0:num_val_samples,:]
        for ridge in ridges:
            Wo=torch.linalg.solve(G_val+ridge*torch.eye(G_val.size(dim=0)),Q_val).T #better nmerical stability than .inv
            Y_train_pred=Features[num_val_samples::,:]@Wo.T
            losses.append(F.mse_loss(Y_train_pred,Y[num_val_samples::,:]))
        ridge=ridges[np.argmin(np.array(losses))]
        logging.info("Optimal lambda: "+str(ridge))
        return ridge

    def after_task(self):
        self._known_classes = self._classes_seen_so_far