import argparse
import os
import os.path
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
from torch.nn import DataParallel
import tqdm
import yaml

from models import STGCNAction
from data_provider import DatasetJSON
from utils import save_model, load_model, load_pretrained_model
import time
import coremltools as ct


INPUT_CHANNELS = 3  # [x, y, confidence]

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def run(args):
    start_time = time.time()
    use_cuda = True
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        print('trainng on CUDA devices')
        logging.info("trainng on CUDA devices")
    else:
        args.device = torch.device('cpu')
        use_cuda = False
        print('training on CPU')
        logging.info("training on CPU")

    if args.export_only:
        export_pytorch_to_coreml(args)
        return

    # setup model and loss
    model = STGCNAction(INPUT_CHANNELS, args.num_keypoints, args.num_actions, args.dropout)
    loss_function = nn.CrossEntropyLoss()

    model.apply(weights_init)
    if args.resume:
        print("Resume from model: ", args.resume)
        logging.info('Resume from model: %s' %(args.resume))
        if args.pretrained:
            load_pretrained_model(args.resume, model, ['torso.linear.weight', 'torso.linear.bias'], args.device)
        else:
            load_model(args.resume, model, args.device)

    model.to(args.device)

    if use_cuda:
        model = DataParallel(model.cuda())

    loss_function.to(args.device)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, weight_decay=0.001, momentum=0.9, nesterov=True) #weight_decay original 0.001
    
    print("loading data")
    logging.info("loading data")
    # testing use all available frames, use batch size of 1
    testing_data = DatasetJSON(path=args.dataset, split_file=args.dataset_split, partition='val', actions=args.actions, temporal_win=args.time_window, num_keypoints=args.num_keypoints, input_channels=INPUT_CHANNELS, sample_interval=args.sample_interval, random_sample=args.random_sample, dataset_cfg=args.dataset_cfg, uniform_sample=args.uniform_sample, slack_len=args.slack_len)
    
    # testing use all available frames, use batch size of 1
    test_batch_size = args.train_batch #1
    test_data_loader = torch.utils.data.DataLoader(testing_data, batch_size=test_batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=args.workers)
    print("loaded data : testing (%d)" %(len(testing_data)))
    logging.info("loaded data : testing (%d)" %(len(testing_data)))
    if args.test_only:
        confusion_meter, test_accuracy = test(test_data_loader, model, args.device, args)
        print('testing accuracy %f' %(test_accuracy))
        logging.info('testing accuracy %f' %(test_accuracy))
        return

    training_data = DatasetJSON(path=args.dataset, split_file=args.dataset_split, partition="train", actions=args.actions, temporal_win=args.time_window, num_keypoints=args.num_keypoints, input_channels=INPUT_CHANNELS, sample_interval=args.sample_interval, random_sample=args.random_sample, dataset_cfg=args.dataset_cfg, uniform_sample=args.uniform_sample, slack_len=args.slack_len)
    train_data_loader = torch.utils.data.DataLoader(training_data, batch_size=args.train_batch, shuffle=True, drop_last=False, pin_memory=True, num_workers=args.workers)
    print("loaded data : training (%d)" %(len(training_data)))
    logging.info("loaded data : training (%d)" %(len(training_data)))

    print("Time elapsed: %4.4f" % (time.time() - start_time))
    logging.info("Time elapsed: %4.4f" % (time.time() - start_time))

    best_acc = 0.0
    best_train_acc = 0.0
    no_improve_epoch = 0
    for epoch in range(args.epochs):
        # adjust learning rate
        adjusted, lr = adjust_lr(args.steps, args.base_lr, optimizer, epoch)
        if adjusted:
            print('adjust learning rate to %f' %lr)
            logging.info('adjust learning rate to %f' %lr)
        
        startTrainTime = time.time()
        train_accuracy= train(train_data_loader, model, args.device, loss_function, optimizer, args)
        endTrainTime = time.time()
        print("One unit train time: ", endTrainTime - startTrainTime)

        confusion_meter, test_accuracy, failed_paths_test = test(test_data_loader, model, args.device, args)
        endTestTime = time.time()
        print("One unit validate time: ", endTestTime - endTrainTime)
        print("One unit total time: ", endTestTime - startTrainTime)

        print('epoch %d : training accuracy %f, testing accuracy %f, time elapsed %4.4f' %(epoch, train_accuracy, test_accuracy, time.time() - start_time))
        logging.info('epoch %d : training accuracy %f, testing accuracy %f, time elapsed %4.4f' %(epoch, train_accuracy, test_accuracy, time.time() - start_time))

        # save model
        if (test_accuracy >= best_acc):
            logging.info(f"failed paths test: {failed_paths_test}")
            best_acc = test_accuracy
            no_improve_epoch = 0
            if (test_accuracy > 50) or (train_accuracy > best_train_acc and test_accuracy > 85):
                print('Confusion matrix: test')
                logging.info("Confusion matrix: test")
                print(confusion_meter.value())
                logging.info(confusion_meter.value())
                model_name = 'checkpoint_epoch%d_val%d_train%d.pt' %(epoch, int(round(test_accuracy)), int(round(train_accuracy)))
                if not os.path.exists(args.experiment):
                    os.makedirs(args.experiment)
                save_model(model, os.path.join(args.experiment, model_name), epoch, optimizer, test_accuracy)
        else:
            no_improve_epoch += 1
            print("no improve epochs: ", no_improve_epoch, "best val accuracy: ", best_acc)
            logging.info('no improve epochs: %d, best val accuracy: %f' %(no_improve_epoch, best_acc))
        
        if train_accuracy >= best_train_acc:
            best_train_acc = train_accuracy

        if no_improve_epoch > args.patiences:
            print("stop training....")
            logging.info("stop training....")
            break

def cal_top1(confusion_matrix, num_actions):
    total = 0
    fp = 0
    for idx in range (num_actions):
        fp += confusion_matrix.value()[idx][idx]
        total += sum(confusion_matrix.value()[idx])
    return 100.0 * fp / total

def adjust_lr(steps, base_lr, optimizer, epoch):
    lr = base_lr
    adjusted = False
    for step in steps:
        if epoch >= step:
            lr *= 0.1
        if epoch == step:
            adjusted = True
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return adjusted, lr

def train(data_loader, model, device, loss_function, optimizer, args):
    # setup confusion meter
    confusion_meter = tnt.meter.ConfusionMeter(args.num_actions, normalized=True)
    confusion_meter.reset()
    # training
    model.train()

    show_range = range(len(data_loader))
    show_range = tqdm.tqdm(show_range, total=len(data_loader), ascii=True)
    batchCount = 0
    accuLoss = 0
    for keypoints, targets, file_paths in data_loader:
        show_range.update(1)

        model.zero_grad()
        targets = targets.to(device)
        results = model(keypoints.to(device))
        loss = loss_function(results, targets)
        batchCount = batchCount + 1
        accuLoss = accuLoss + loss
        
        loss.backward()
        optimizer.step()
        del loss
        
        # add to confusion meter
        confusion_meter.add(results.detach(), targets.detach())
    
    print("train avgLoss: ", accuLoss / batchCount)
    # return top1 accuracy
    return cal_top1(confusion_meter, args.num_actions)

def test(data_loader, model, device, args):
    # setup confusion meter
    confusion_meter = tnt.meter.ConfusionMeter(args.num_actions, normalized=False)
    confusion_meter.reset()
    # testing 
    model.eval() # this will turn off dropout
    failed_paths = []
    for keypoints, targets, file_paths in data_loader:
        print(keypoints.shape)
        print(targets)
        print(file_paths)
        exit()
        targets = targets.to(device)
        results = model(keypoints.to(device))
        confusion_meter.add(results.detach(), targets.detach())
    return confusion_meter, cal_top1(confusion_meter, args.num_actions), failed_paths

def export_pytorch_to_coreml(args):
    # load pytorch model
    model = STGCNAction(INPUT_CHANNELS, args.num_keypoints, args.num_actions, args.dropout, export=True)
    if args.resume:
        load_model(args.resume, model, args.device)
    model.eval()
    # trace
    input_shape = [args.time_window, INPUT_CHANNELS, args.num_keypoints]
    data = torch.zeros(input_shape, dtype=torch.float32)
    traced_model = torch.jit.trace(model, data)
    # convert
    class_labels = args.actions
    model = ct.convert(
        traced_model, 
        inputs = [ct.TensorType(name="poses", shape=input_shape)], # pytorch ct does not work with "input" as name
        classifier_config = ct.ClassifierConfig(class_labels)
    )
    # modify the input dimension to be compatible with Vision API
    spec = model.get_spec()
    
    # set new input
    spec.description.ClearField("input")
    first_input = spec.description.input.add()
    first_input.name = "poses"
    reshaped_input_shape = [args.time_window, INPUT_CHANNELS, args.num_keypoints]
    first_input.type.multiArrayType.shape.extend(reshaped_input_shape)
    first_input.type.multiArrayType.dataType = 65568 # FLOAT_32
    
    # save the spec to coreml
    ct.utils.rename_feature(spec, '348', 'labelProbabilities') # rename the output probability name ('536' output name changes sometimes! update it if errors occur)
    ct.utils.rename_feature(spec, 'classLabel', 'label') 
    model = ct.models.MLModel(spec)
     # modify metadata
    model.input_description["poses"] = "Input hand poses to be classified."
    model.output_description["classLabel"] = "Most likely hand gesture category."
    model.output_description["labelProbabilities"] = "A dictionary of class labels and the corresponding confidences."
    model.author = ""
    model.license = "MIT"
    model.short_description = "Recognize a static or dynamic hand gesture from a single or a series of hand poses."
    model.version = "1.0"
    # save the coreml model file
    model.save("HandGestureClassifier.mlmodel")



def parse_command_line():
    parser = argparse.ArgumentParser(description='(PyTorch) training gcn-based action recognition')
    parser.add_argument('-c', '--config', default=None, help='path to yaml config file')
    parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--dataset', type=str, default='./data', help='dataset path that should contain a list of JSON keypoint files')
    parser.add_argument('--dataset_split', type=str, default='./data', help='dataset split JSON file path')
    parser.add_argument('--log_file', type=str, default=None, help='log file to save the training report')
    parser.add_argument('--dataset_cfg', default=0, type=int, help='dataset config id. 1 for DHG that label id starts from 1; 0 for hand emoji that label id starts from 0.')
    parser.add_argument('--train_batch', default=16, type=int, help='batchsize for training')
    parser.add_argument('--base_lr', '--base_learning_rate', default=1e-2, type=float, help='base learning rate')
    parser.add_argument('--dropout', default=0.0, type=float, help='droput')
    parser.add_argument('--epochs', default=90, type=int, help='number of total epochs to run')
    parser.add_argument('--workers', default=1, type=int, help='number of data loading workers (default: 1)')
    parser.add_argument('--export_only', action='store_true', help='export action model to pytorch format')
    parser.add_argument('-r', '--resume', type=str, default=None, help='resume train/test with model')
    parser.add_argument('-e', '--experiment', type=str, default='test', help='epochs interval for savin model')
    parser.add_argument('--pretrained', action='store_true', help='loaded pretrained model by ignoring last fcn layer')
    parser.add_argument('--num_keypoints', type=int, default=21, help='number of keypoints per person')
    parser.add_argument('--time_window', type=int, default=32, help='number of frames to be used in a time window')
    parser.add_argument('--slack_len', type=int, default=0, help='number of slack frames used before and after a valid annotation, e.g., only appen_dynamic dataset uses this, other datasets not.')
    parser.add_argument('--num_actions', type=int, default=14, help='num of actions')
    parser.add_argument('--patiences', type=int, default=50, help='number of no-improvement epochs allowed before stop training.')
    parser.add_argument('--sample_interval', type=int, default=1, help='sample interval on input videos. 1 for 30fps, 2 for 60 fps.')
    parser.add_argument('--test_only', action='store_true', help='testing only')
    parser.add_argument('--random_sample', action='store_true', help='testing with random sample, only useful when sample_interval>1.')
    parser.add_argument('--uniform_sample', action='store_true', help='train with random sampled frames that distribute uniformly across the entire video. False if consecutive frames are used.')
    parser.add_argument('--steps', type=int, default=[], nargs='+', help='steps for learning rate adjustment')
    parser.add_argument('--actions', type=str, default=[], nargs='+', help='action types')
    temp_args = parser.parse_args()

    if temp_args.config:
        with open(temp_args.config, 'r') as f:
            default_arg = yaml.safe_load(f)
            parser.set_defaults(**default_arg)
    args = parser.parse_args()
    

    log_file = "report_results/logs.txt" # default temp log file if path is not provided
    if args.log_file != None:
        log_file = args.log_file

    for handler in logging.root.handlers[:]: # save the logger issue that info is not going into the file sometimes
        logging.root.removeHandler(handler)

    logging.basicConfig(level=logging.DEBUG, filename=log_file, filemode="w", format="%(asctime)-15s %(levelname)-8s %(message)s")
    print("Save training logs to file %s" %(log_file))
    
    print(args)
    logging.info(args)
    return args


if __name__ == '__main__':
    args = parse_command_line()
    run(args)
   
    
