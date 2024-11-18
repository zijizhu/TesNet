import os
import shutil

import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re

from util.helpers import makedir
import push, model, train_and_test as tnt
from util import save
from util.log import create_logger
from util.preprocess import mean, std, preprocess_input_function

import settings_CUB
import settings_cars
import settings_dogs
from util.data import DogsDataset

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid',type=str, default='0')
parser.add_argument('-arch',type=str, default='vgg19')
parser.add_argument('-num_prototypes', type=int, default=2000)

parser.add_argument('-dataset',type=str,default="CUB", choices=["CUB", "Cars", "Dogs"])
parser.add_argument('-times',type=str,default="test",help="experiment_run")

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
print(os.environ['CUDA_VISIBLE_DEVICES'])


#setting parameter
experiment_run = args.times
base_architecture = args.arch
dataset_name = args.dataset

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)
#model save dir
model_dir = './saved_models/' + dataset_name+'/' + base_architecture + '_' + str(args.num_prototypes) + '/'

if os.path.exists(model_dir) is True:
    shutil.rmtree(model_dir)
makedir(model_dir)


shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'settings_CUB.py'), dst=model_dir)
# shutil.copy(src=os.path.join(os.getcwd(), 'models', base_architecture_type + '_features.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'


# load the hyper param
if dataset_name == "CUB":
    #model param
    num_classes = settings_CUB.num_classes
    img_size = settings_CUB.img_size
    add_on_layers_type = settings_CUB.add_on_layers_type
    prototype_shape = settings_CUB.prototype_shape
    prototype_shape = (args.num_prototypes, prototype_shape[1], prototype_shape[2], prototype_shape[3],)

    prototype_activation_function = settings_CUB.prototype_activation_function
    #datasets
    train_dir = settings_CUB.train_dir
    test_dir = settings_CUB.test_dir
    train_push_dir = settings_CUB.train_push_dir
    train_batch_size = settings_CUB.train_batch_size
    test_batch_size = settings_CUB.test_batch_size
    train_push_batch_size = settings_CUB.train_push_batch_size
    #optimzer
    joint_optimizer_lrs = settings_CUB.joint_optimizer_lrs
    joint_lr_step_size = settings_CUB.joint_lr_step_size
    warm_optimizer_lrs = settings_CUB.warm_optimizer_lrs

    last_layer_optimizer_lr = settings_CUB.last_layer_optimizer_lr
    # weighting of different training losses
    coefs = settings_CUB.coefs
    # number of training epochs, number of warm epochs, push start epoch, push epochs
    num_train_epochs = settings_CUB.num_train_epochs
    num_warm_epochs = settings_CUB.num_warm_epochs
    push_start = settings_CUB.push_start
    push_epochs = settings_CUB.push_epochs
elif dataset_name == "Cars":
    num_classes = settings_cars.num_classes
    img_size = settings_cars.img_size
    add_on_layers_type = settings_cars.add_on_layers_type
    prototype_shape = settings_cars.prototype_shape
    prototype_shape = (args.num_prototypes, prototype_shape[1], prototype_shape[2], prototype_shape[3],)

    prototype_activation_function = settings_cars.prototype_activation_function
    #datasets
    data_dir = settings_cars.train_dir
    train_batch_size = settings_cars.train_batch_size
    test_batch_size = settings_cars.test_batch_size
    train_push_batch_size = settings_cars.train_push_batch_size
    #optimzer
    joint_optimizer_lrs = settings_cars.joint_optimizer_lrs
    joint_lr_step_size = settings_cars.joint_lr_step_size
    warm_optimizer_lrs = settings_cars.warm_optimizer_lrs

    last_layer_optimizer_lr = settings_cars.last_layer_optimizer_lr
    # weighting of different training losses
    coefs = settings_cars.coefs
    # number of training epochs, number of warm epochs, push start epoch, push epochs
    num_train_epochs = settings_cars.num_train_epochs
    num_warm_epochs = settings_cars.num_warm_epochs
    push_start = settings_cars.push_start
    push_epochs = settings_cars.push_epochs
elif dataset_name == "Dogs":
    num_classes = settings_dogs.num_classes
    img_size = settings_dogs.img_size
    add_on_layers_type = settings_dogs.add_on_layers_type
    prototype_shape = settings_dogs.prototype_shape
    prototype_shape = (args.num_prototypes, prototype_shape[1], prototype_shape[2], prototype_shape[3],)

    prototype_activation_function = settings_dogs.prototype_activation_function
    #datasets
    data_dir = settings_dogs.train_dir
    train_batch_size = settings_dogs.train_batch_size
    test_batch_size = settings_dogs.test_batch_size
    train_push_batch_size = settings_dogs.train_push_batch_size
    #optimzer
    joint_optimizer_lrs = settings_dogs.joint_optimizer_lrs
    joint_lr_step_size = settings_dogs.joint_lr_step_size
    warm_optimizer_lrs = settings_dogs.warm_optimizer_lrs

    last_layer_optimizer_lr = settings_dogs.last_layer_optimizer_lr
    # weighting of different training losses
    coefs = settings_dogs.coefs
    # number of training epochs, number of warm epochs, push start epoch, push epochs
    num_train_epochs = settings_dogs.num_train_epochs
    num_warm_epochs = settings_dogs.num_warm_epochs
    push_start = settings_dogs.push_start
    push_epochs = settings_dogs.push_epochs
else:
    raise Exception("there are no settings file of datasets {}".format(dataset_name))

normalize = transforms.Normalize(mean=mean,std=std)

# all datasets
# train set
if args.dataset == "CUB":
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
    # push set
    train_push_dataset = datasets.ImageFolder(
        train_push_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
        ]))
    # test set
    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
elif args.dataset == "Cars":
    train_dataset = datasets.StanfordCars(
        data_dir, split="train", download=False,
        transform= transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_push_dataset = datasets.StanfordCars(
        data_dir, split="train", download=False,
        transform= transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])
    )
    test_dataset = datasets.StanfordCars(
        args.data_path, split="test", download=False,
        transform= transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])
    )
elif args.dataset == "Dogs":
    train_dataset = DogsDataset(
        root="datasets", split="train",
        transform= transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_push_dataset = DogsDataset(
        root="datasets", split="train",
        transform= transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])
    )
    test_dataset = DogsDataset(
        root="datasets", split="test",
        transform= transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])
    )

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=4, pin_memory=False)

train_push_loader = torch.utils.data.DataLoader(
        train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
        num_workers=4, pin_memory=False)
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=4, pin_memory=False)

# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

log("backbone architecture:{}".format(base_architecture))
log("basis concept size:{}".format(prototype_shape))
# construct the model
ppnet = model.construct_TesNet(base_architecture=base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type)
#if prototype_activation_function == 'linear':
#    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)


class_specific = True

# define optimizer
from settings_CUB import joint_optimizer_lrs, joint_lr_step_size
joint_optimizer_specs = \
[{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
 {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

from settings_CUB import warm_optimizer_lrs
warm_optimizer_specs = \
[{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

from settings_CUB import last_layer_optimizer_lr
last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

#best acc
best_acc = 0
best_epoch = 0
best_time = 0

# train the model
log('start training')
max_accu = 0.0
for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))
    #stage 1: Embedding space learning
    #train
    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log)
        _,train_results = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)
    else:
        tnt.joint(model=ppnet_multi, log=log)
        joint_lr_scheduler.step()
        _,train_results = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)

    #test
    accu,test_results = tnt.test(model=ppnet_multi, dataloader=test_loader,
                    class_specific=class_specific, log=log)

    if accu >= max_accu:
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                    target_accu=0.70, log=log)
        max_accu = accu
    #stage2: Embedding space transparency
    if epoch >= push_start and epoch in push_epochs:
        push.push_prototypes(
            train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
            epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        accu,test_results = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log)
        if accu >= max_accu:
            if args.dataset == "CUB":
                save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                            target_accu=0.70, log=log)
            max_accu = accu
    #stage3: concept based classification
        if prototype_activation_function != 'linear':
            tnt.last_only(model=ppnet_multi, log=log)
            for i in range(20):
                log('iteration: \t{0}'.format(i))
                _,train_results= tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                              class_specific=class_specific, coefs=coefs, log=log)

                accu,test_results = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                class_specific=class_specific, log=log)
                if accu >= max_accu:
                    if args.dataset == "CUB":
                        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                                    target_accu=0.70, log=log)
                    max_accu = accu
   
logclose()

