import torch
import random
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_data(data):

    inputs = tokenizer.batch_encode_plus([sample['text'] for sample in data],\
                                        add_special_tokens=True,\
                                        max_length=512,\
                                        return_tensors='pt',\
                                        padding='max_length',\
                                        truncation=True)

    return inputs

tokenizer = BertTokenizer.from_pretrained(\
        pretrained_model_name_or_path='/home/charles/code/bert-base-cased')

model = BertForSequenceClassification.from_pretrained(\
        pretrained_model_name_or_path='/home/charles/code/bert-base-cased',\
        num_labels=2)

model.to(device)

f = open('/home/charles/code/my_json.json','r')

content = f.read()

train_data = json.loads(content)

train_inputs = preprocess_data(train_data)

train_labels = torch.tensor([sample['label'] for sample in train_data])

train_dataset = torch.utils.data.TensorDataset(train_inputs['input_ids'],\
                                               train_inputs['attention_mask'],\
                                               train_labels)

train_dataloader = DataLoader(train_dataset,\
                              batch_size=8,\
                              shuffle=True)

optimizer = AdamW(model.parameters(), lr=1e-5)

def train(model, train_dataloader, optimizer):

    model.train()

    total_loss = 0

    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
        outputs = model(**inputs)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    return total_loss

num_epochs = 50

def predict(model,train_data):

    resultList = random.sample(range(0,3500),100) 

    test_data = []

    test_lab = []

    for i in resultList:
        test_data.append({'text':train_data[i]['text']})
        test_lab.append(train_data[i]['label'])

    tmp_i,totl = 0,0

    model.eval()

    inputs = preprocess_data(test_data)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()

    for itm in probs:
        x1,x2 = itm[0],itm[1]
        if x1 > x2:
            if test_lab[tmp_i]==0:
                totl = totl+1
        else:
            if test_lab[tmp_i]==1:
                totl = totl+1
        tmp_i = tmp_i + 1

    return totl

for epoch in range(num_epochs):

    total_loss = train(model, train_dataloader, optimizer)

    print(f"Epoch {epoch+1}: Loss = {total_loss} accu = {predict(model,train_data)}")

torch.save(model.state_dict(),'outputdr')
