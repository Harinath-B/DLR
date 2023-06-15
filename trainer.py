import torch
import time
import random

def getRandom(min, max):
    return random.randint(min, max)

def train_slr(model_s, trainloader, testloader, criterion, optimizer, num_epochs=10):
    model_s = model_s.cuda()
    slr_epoch_acc_output = []
    slr_epoch_test_acc_output = []
    start_time = time.time()
    for epoch in range(0, num_epochs):
        print(f"Epoch: {epoch}")
        train_correct = 0.
        train_total = 0.
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()

            outputs = model_s(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Check for our extra reinforcement
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum()     

            optimizer.step()

            # Print stats
            running_loss += loss.item()

        # Test the Network
        test_correct = 0.
        test_total = 0.
        with torch.no_grad():
            for data in testloader:
                test_images, test_labels = data
                test_images, test_labels = test_images.cuda(), test_labels.cuda()
                test_outputs = model_s(test_images)
                _, test_predicted = torch.max(test_outputs.data, 1)
                test_total += test_labels.size(0)
                test_correct += (test_predicted == test_labels).sum().item()

                c_test = (test_predicted == test_labels).squeeze()
                if (len(c_test.size()) == 0):
                    continue
                
        train_acc = (100. * train_correct.item() / train_total)
        print("Accuracy of the network for this batch: %.4f %%" % (train_acc))
        slr_epoch_acc_output.append(train_acc)
                
        test_acc = (100. * test_correct / test_total)
        slr_epoch_test_acc_output.append(test_acc)
        # lr_output.append(lr)


    end_time = time.time()
    
    return {'time': (end_time - start_time), 
            'train_acc': slr_epoch_acc_output,
            'test_acc': slr_epoch_test_acc_output}
    
def train_dlr(model_d, trainloader, testloader, criterion, optimizer, batch_size, params, num_epochs=10):
    correct_learning_rate = params['cor_lr']
    incorrect_learning_rate = params['incor_lr']
    cor_lr_change = params['cor_lr_change']
    incor_lr_change = params['incor_lr_change']
    
    dlr_epoch_acc_output = []
    dlr_epoch_test_acc_output = []
    cor_lr_output = []
    incor_lr_output = []
    cor_min_rate = params['cor_min']
    cor_max_rate = params['cor_max']
    incor_min_rate = params['incor_min']
    incor_max_rate = params['incor_max']
    start_time = time.time()
    for epoch in range(0, num_epochs):
        print(f"Epoch: {epoch}")
        train_correct = 0.
        train_total = 0.
        correct_count = 0
        incorrect_count = 0
        correct_rand_ratio = getRandom(cor_min_rate, cor_max_rate)
        incorrect_rand_ratio = getRandom(incor_min_rate, incor_max_rate)
        running_loss = 0.0
        for i, data in enumerate(trainloader):
                    
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
    
            outputs = model_d(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
    
            # Check for our extra reinforcement
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            current_correct = (predicted == labels).sum()
            train_correct += current_correct
            current_incorrect = batch_size - current_correct
            
            # only issue with doing it this way within batches is that
            # we don't have the most up to date lr for each example.
            # However, the performance increase with batches makes it worthwhile
    
            # Update learning rate based on correct and incorrect responses
            is_correct = 0
            if current_correct >= current_incorrect:
                correct_count += current_correct
                incorrect_count += current_incorrect
                is_correct = 1
            else:
                correct_count += current_correct
                incorrect_count += current_incorrect
                is_correct = 0
    
            if correct_count >= correct_rand_ratio:
                correct_learning_rate = correct_learning_rate - cor_lr_change
                correct_count = 0
                correct_rand_ratio = getRandom(int(cor_min_rate), int(cor_max_rate))
                
                if correct_learning_rate < 0:
                    correct_learning_rate = 0.000001
    
            if incorrect_count >= incorrect_rand_ratio:
                incorrect_learning_rate = incorrect_learning_rate + incor_lr_change
                incorrect_count = 0
                incorrect_rand_ratio = getRandom(int(incor_min_rate), int(incor_max_rate))
                if incorrect_learning_rate < 0:
                    incorrect_learning_rate = 0.000001
    
    
            for param_group in optimizer.param_groups:
                if is_correct: 
                    param_group['lr'] = correct_learning_rate
                else:
                    param_group['lr'] = incorrect_learning_rate
            
            optimizer.step()
    
            # Print stats
            running_loss += loss.item()
    
        # Test the Network
        test_correct = 0.
        test_total = 0.
        with torch.no_grad():
            for data in testloader:
                test_images, test_labels = data
                test_images, test_labels = test_images.cuda(), test_labels.cuda()
                test_outputs = model_d(test_images)
                _, test_predicted = torch.max(test_outputs.data, 1)
                test_total += test_labels.size(0)
                test_correct += (test_predicted == test_labels).sum().item()
    
                c_test = (test_predicted == test_labels).squeeze()
                if (len(c_test.size()) == 0):
                    continue
                
        test_acc = (100. * test_correct / test_total)
        dlr_epoch_test_acc_output.append(test_acc)
        cor_lr_output.append(correct_learning_rate)
        incor_lr_output.append(incorrect_learning_rate)
        train_acc = (100. * train_correct.item() / train_total)
        print("Accuracy of the network for this batch: %.4f %%" % (train_acc))
        dlr_epoch_acc_output.append(train_acc)
    end_time = time.time()
    
    return {'time': (end_time - start_time), 
            'train_acc': dlr_epoch_acc_output,
            'test_acc': dlr_epoch_test_acc_output,
            'cor_lr': correct_learning_rate,
            'incor_lr': incorrect_learning_rate}