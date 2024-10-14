import numpy as np
import torch
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report

from datasets import load_metric

import utils


def evaluate(model, criterion, dataloader, task_running='task-1', cm=False, cr=False, last_epoch=False, device='cpu'):
    if task_running == 'task-1':
        dimesion = 1
    elif task_running == 'task-2':
        dimesion = 2

    running_loss = 0.0
    predictions = []
    true_labels = []

    model.eval()
    with torch.no_grad():
        # Wrap the dataloader with tqdm to monitor progress
        with tqdm(dataloader, desc="Evaluation") as tqdm_loader:
            for batch_idx, batch in enumerate(tqdm_loader):
                tqdm_loader.set_description(f"Evaluation, Batch {batch_idx + 1}/{len(dataloader)}")

                inputs = batch['input'].to(device)
                label = batch['label'].to(device)

                outputs = model(inputs)

                if task_running == 'task-2':
                    loss1 = criterion(outputs[0], label[:, 0, :]) # title
                    loss2 = criterion(outputs[1], label[:, 1, :]) # desc
                    loss3 = criterion(outputs[2], label[:, 2, :]) # comp
                    loss4 = criterion(outputs[3], label[:, 3, :]) # other
                    loss = loss1 + loss2 + loss3 + loss4

                    outputs = torch.stack(outputs)
                    outputs = outputs.transpose(0, 1)

                elif task_running == 'task-1':
                    loss = criterion(outputs, label)

                running_loss += loss.item()

                _, pred = torch.max(outputs, dim=dimesion)
                _, true = torch.max(label, dim=dimesion)

                # Append predictions and true labels for F1 score calculation
                predictions.extend(pred.tolist())
                true_labels.extend(true.tolist())

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    print(f'Evaluation, Loss: {running_loss:.4f}, ', end="")
    
    if task_running == 'task-1':
        show_evaluation_task_1(true_labels, predictions)
        if cm and cr:
            show_cm_cr_task_1(true_labels, predictions)
    elif task_running == 'task-2':
        show_evaluation_task_2(true_labels, predictions)
        if cm and cr:
            show_cm_cr_task_2(true_labels, predictions)

    return predictions, true_labels, running_loss


def train(model, criterion, optimizer, epochs, train_dataloader, dev_dataloader, saving_path=None, task_running='task-1', device='cpu'):
    
    if task_running == 'task-1':
        dimesion = 1
    elif task_running == 'task-2':
        dimesion = 2
    
    patience = 5
    patience_counter = 0
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        true_labels = []
        predictions = []

        # Wrap the train_dataloader with tqdm to monitor progress
        with tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}") as tqdm_loader:
            for batch_idx, batch in enumerate(tqdm_loader):
                tqdm_loader.set_description(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_dataloader)}")

                inputs = batch['input'].to(device)
                label = batch['label'].to(device)

                optimizer.zero_grad()
                outputs = model(inputs)

                if task_running == 'task-2':
                    loss1 = criterion(outputs[0], label[:, 0, :]) # title
                    loss2 = criterion(outputs[1], label[:, 1, :]) # desc
                    loss3 = criterion(outputs[2], label[:, 2, :]) # comp
                    loss4 = criterion(outputs[3], label[:, 3, :]) # other
                    loss = loss1 + loss2 + loss3 + loss4

                    loss.backward()
                    optimizer.step()

                    outputs = torch.stack(outputs)
                    outputs = outputs.transpose(0, 1)

                elif task_running == 'task-1':
                    loss = criterion(outputs, label)
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()

                _, pred = torch.max(outputs, dim=dimesion)
                _, true = torch.max(label, dim=dimesion)
                
                # Append predictions and true labels for metrics calculation
                true_labels.extend(true.tolist())
                predictions.extend(pred.tolist())

        true_labels = np.array(true_labels)
        predictions = np.array(predictions)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss:.4f}, ', end='')

        if task_running == 'task-1':
            show_evaluation_task_1(true_labels, predictions)
        elif task_running == 'task-2':
            show_evaluation_task_2(true_labels, predictions)

        # Evaluation on Dev set
        _, _, dev_running_loss = evaluate(model, criterion, dev_dataloader, cm=False, cr=False, last_epoch=False, task_running=task_running, device=device)

        # Saving
        if dev_running_loss < best_val_loss:
            best_val_loss = dev_running_loss
            patience_counter = 0
            if saving_path:
                path = saving_path + "_" + str(epoch) + '.pth'
                torch.save(model.state_dict(), path)
                print('Saved the best model to path: ' +  path)
        else:
            patience_counter += 1

        # early stopping
        if patience_counter >= patience:
            print('Early stopping triggered')
            break
        
        print()


def show_evaluation_task_1(true_labels, predictions):
    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro')
    prec = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    
    print(
        f'Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}'
    )


def show_cm_cr_task_1(true_labels, predictions):
    class_names = ['clean', 'warning', 'seeding'] # 0, 1, 2
    cm = confusion_matrix(true_labels, predictions)
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(true_labels, predictions, target_names=class_names))


def show_evaluation_task_2(true_labels, predictions):
    accs = []
    precs_1, f1s_1, recalls_1 = [], [], []
    precs_2, f1s_2, recalls_2 = [], [], []
    precs_3, f1s_3, recalls_3 = [], [], []

    for i in range(4):
        accs.append(accuracy_score(true_labels[:, i], predictions[:, i]))

        precs_1.append(precision_score(true_labels[:, i], predictions[:, i], average='macro'))
        f1s_1.append(f1_score(true_labels[:, i], predictions[:, i], average='macro'))
        recalls_1.append(recall_score(true_labels[:, i], predictions[:, i], average='macro'))

        precs_2.append(precision_score(true_labels[:, i], predictions[:, i], average='micro'))
        f1s_2.append(f1_score(true_labels[:, i], predictions[:, i], average='micro'))
        recalls_2.append(recall_score(true_labels[:, i], predictions[:, i], average='micro'))

        precs_3.append(precision_score(true_labels[:, i], predictions[:, i], average='weighted'))
        f1s_3.append(f1_score(true_labels[:, i], predictions[:, i], average='weighted'))
        recalls_3.append(recall_score(true_labels[:, i], predictions[:, i], average='weighted'))

    acc = np.mean(accs)
    
    print()
    print()
    print('accs: ', end='')
    for i in accs: print(f'{i:.4f}, ', end='')
    print(f' => {acc:.4f}')
    print()

    prec1 = np.mean(precs_1)
    recall1 = np.mean(recalls_1)
    f11 = np.mean(f1s_1)

    print('precs (macro): ', end='')
    for i in precs_1: print(f'{i:.4f}, ', end='')
    print(f' => {prec1:.4f}')
    print('recalls (macro): ', end='')
    for i in recalls_1: print(f'{i:.4f}, ', end='')
    print(f' => {recall1:.4f}')
    print('f1s (macro): ', end='')
    for i in f1s_1: print(f'{i:.4f}, ', end='')
    print(f' => {f11:.4f}')
    print()

    prec2 = np.mean(precs_2)
    recall2 = np.mean(recalls_2)
    f12 = np.mean(f1s_2)
    
    print('precs (micro): ', end='')
    for i in precs_2: print(f'{i:.4f}, ', end='')
    print(f' => {prec2:.4f}')
    print('recalls (micro): ', end='')
    for i in recalls_2: print(f'{i:.4f}, ', end='')
    print(f' => {recall2:.4f}')
    print('f1s (micro): ', end='')
    for i in f1s_2: print(f'{i:.4f}, ', end='')
    print(f' => {f12:.4f}')
    print()

    prec3 = np.mean(precs_3)
    recall3 = np.mean(recalls_3)
    f13 = np.mean(f1s_3)
    
    print('precs (weighed): ', end='')
    for i in precs_3: print(f'{i:.4f}, ', end='')
    print(f' => {prec3:.4f}')
    print('recalls (weighed): ', end='')
    for i in recalls_3: print(f'{i:.4f}, ', end='')
    print(f' => {recall3:.4f}')
    print('f1s (weighed): ', end='')
    for i in f1s_3: print(f'{i:.4f}, ', end='')
    print(f' => {f13:.4f}')
    print()


def show_cm_cr_task_2(true_labels, predictions):
    aspect_names = ['title', 'desc', 'company', 'other']
    # class_names = ['neutral', 'positive', 'negative', 'not_mentioned']
    for i, aspect in enumerate(aspect_names):
        cm_p = confusion_matrix(true_labels[:, i], predictions[:, i])
        print(f"Confusion Matrix of {aspect} aspect")
        print(cm_p)
        print(f"Classification Report for {aspect} aspect")
        print(classification_report(true_labels[:, i], predictions[:, i]))#, target_names=class_names))


def generate_task_3(model, tokenizer, dataloader, target_len=512, device='cpu'):

    predictions = []
    references = []

    model.eval()
    with torch.no_grad():
        # Wrap the dataloader with tqdm to monitor progress
        with tqdm(dataloader, desc="Evaluation") as tqdm_loader:
            for batch_idx, data in enumerate(tqdm_loader, 0):
                tqdm_loader.set_description(f"Validation, Batch {batch_idx + 1}/{len(dataloader)}")

                ids = data['input']['input_ids'].to(device, dtype=torch.long)
                mask = data['input']['attention_mask'].to(device, dtype=torch.long)
                y = data['label']['input_ids'].to(device, dtype=torch.long)

                generated_ids = model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    max_length=target_len,
                    num_beams=2,
                    repetition_penalty=2.5,
                    length_penalty=1.0,
                    early_stopping=True
                )

                preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
                target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]

                predictions.extend(preds)
                references.extend(target)
                
    return predictions, references


def train_task_3(model, optimizer, tokenizer, epochs, train_dataloader, dev_dataloader, target_len, saving_path=None, device='cpu'):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Wrap the train_dataloader with tqdm to monitor progress
        with tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}") as tqdm_loader:
            for batch_idx, batch in enumerate(tqdm_loader):
                tqdm_loader.set_description(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_dataloader)}")

                ids = batch['input']['input_ids'].to(device, dtype=torch.long)
                mask = batch['input']['attention_mask'].to(device, dtype=torch.long)

                y = batch['label']['input_ids'].to(device, dtype=torch.long)
                y_ids = y[:, :-1].contiguous()
                lm_labels = y[:, 1:].clone().detach()
                lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100

                outputs = model(
                    input_ids=ids, attention_mask=mask,
                    decoder_input_ids=y_ids, labels=lm_labels
                )
                loss = outputs[0]
                running_loss += loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss:.4f}')
        
        predictions, references = generate_task_3(model, tokenizer, dev_dataloader, target_len=target_len, device=device)
        bertscore, bleuscore, rougescore = compute_score_task_3(predictions, references)

        print('Evaluation on dev set:')
        print(f'BERT score (prec, rec, f1): {bertscore}')
        print(f'Bleu score (bleu, prec1, prec2, prec3, prec4): {bleuscore}')
        print(f'Rouge score (1, 2, L): {rougescore}')

        if saving_path and epoch == epochs - 1:
            path = saving_path + "_" + str(epoch) + '.pth'
            torch.save(model.state_dict(), path)
            print('Saved the model to path: ' +  path)
            print()


def compute_score_task_3(predictions, references):
    bertscore_metric = load_metric('bertscore')
    bleu_metric = load_metric('bleu')
    rouge_metric = load_metric('rouge')

    bertscore_result = bertscore_metric.compute(predictions=predictions, references=references, lang="vi")
    bertscore_precision = round(np.mean(bertscore_result['precision']), 4)
    bertscore_recall = round(np.mean(bertscore_result['recall']), 4)
    bertscore_f1 = round(np.mean(bertscore_result['f1']), 4)

    bleuscore_result = bleu_metric.compute(
        predictions=[pred.split() for pred in predictions],
        references=[[ref.split()] for ref in references],
    )
    bleuscore = round(bleuscore_result['bleu'], 4)
    bleu_prec_1 = round(bleuscore_result['precisions'][0])
    bleu_prec_2 = round(bleuscore_result['precisions'][1])
    bleu_prec_3 = round(bleuscore_result['precisions'][2])
    bleu_prec_4 = round(bleuscore_result['precisions'][3])

    rouge_result = rouge_metric.compute(predictions=predictions, references=references)
    rouge_1 = round(rouge_result['rouge1'].mid.fmeasure, 4)
    rouge_2 = round(rouge_result['rouge2'].mid.fmeasure, 4)
    rouge_L = round(rouge_result['rougeL'].mid.fmeasure, 4)
    
    return (bertscore_precision, bertscore_recall, bertscore_f1), \
           (bleuscore, bleu_prec_1, bleu_prec_2, bleu_prec_3, bleu_prec_4), \
           (rouge_1, rouge_2, rouge_L)
