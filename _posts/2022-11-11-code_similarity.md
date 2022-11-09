---
layout: post
title: Code Similarity (CODEBERT)
---

# Code Similarity <br>

# Title <br>
<br>

## 코드 유사성 판단이란? <br>
자연어처리 분야의 발달과 함께 사람이 사용하는 자연어가 아닌 프로그래밍 언어를 이해하는 딥러닝 모델에 대한 연구가 진행되고 있다.
코드 유사성 판단은 주어진 두 코드간 유사성(동일한 결과물을 산출하는 코드인지) 여부를 판단하는 것으로 코드의 전체적인 구조와 의미를 딥러닝 모델이 이해하고 있는지 파악할 수 있는 분야이다.<br>


## 데이터 설명 <br>
딥러닝 모델을 학습시킬 데이터셋으로 DACON에서 공개한 [코드 유사성 판단 데이터셋](https://dacon.io/competitions/official/235900/data)을 선정하였다. 데이터셋 파일을 다운로드 받아서 압축을 해제하면 sample_train.csv, test.csv, sample_submission.csv, code (Folder) 이렇게 4개의 파일 및 폴더가 존재하는 것을 확인할 수 있다. 여기서 sample_submission.csv 파일과 test.csv 파일은 대회에 제출하기 위한 레이블이 없는 test 데이터셋과 제출 양식에 관한 파일이다. 이번 포스팅에서는 sample_train.csv 파일에 존재하는 pair들에 대해서 모델을 학습 및 평가를 진행하였다.


## 환경 설정 및 데이터 전처리<br>
먼저 seed와 학습에 사용될 device를 설정하였다. <br>

```
import torch
import random
import numpy as np
import pandas as pd

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(0)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device('cpu')
```

그리고 csv 파일에서 데이터를 읽어와서 학습에 쓰일 데이터와 평가에 쓰일 데이터로 분할하였다. 학습 및 평가 데이터의 비율은 8:2로 구성하였다. 그 외에 주석제거 등의 추가적인 전처리는 진행하지 않았다. <br>

```
data_csv = pd.read_csv("data/sample_train.csv")
data_csv.head()

# # 읽어온 데이터를 code1, code2, similar로 분리
data_code1 = [data_csv["code1"][idx] for idx in range(len(data_csv))]
data_code2 = [data_csv["code2"][idx] for idx in range(len(data_csv))]
data_similar = [int(data_csv["similar"][idx]) for idx in range(len(data_csv))]

n_train = int(len(data_similar)*0.8)
n_test = len(data_similar) - n_train

index_list = list(range(len(data_similar)))
random.shuffle(index_list)

train_code1 = [data_code1[idx] for idx in index_list[:n_train]]
train_code2 = [data_code2[idx] for idx in index_list[:n_train]]
train_similar = [data_similar[idx] for idx in index_list[:n_train]]

test_code1 = [data_code1[idx] for idx in index_list[n_train:]]
test_code2 = [data_code2[idx] for idx in index_list[n_train:]]
test_similar = [data_similar[idx] for idx in index_list[n_train:]]
```

## 모델 구현 <br>
마이크로소프트에서 공개한 [CodeBERT 모델](https://github.com/microsoft/CodeBERT)을 통하여 두 코드의 유사도를 계산하는 모델을 구현하였다. CodeBERT에 대해서 더 자세한 내용은 공식 github를 참고하면 된다. 이 포스팅에서 구현한 모델은 한 쌍의 코드가 sep token으로 구분되어 입력으로 들어가면 두 코드의 유사성을 반환해주는 구조이다. CodeBERT의 CLS Token과 하나의 Linear layer를 통하여 유사성을 계산할 것이며, Linear Layer의 경우 output 차원을 2로 두어 두 코드가 유사하지 않은 경우 outut vector의 index 0의 값이 크도록, 유사한 경우 index 1의 값이 크도록 모델을 구성하였다. huggingface에서 제공하는 tranformers라는 라이브러리를 활용하면 자연어처리에서 주로 사용하는 모델 구조나 함수, 그리고 학습된 모델 파일을 쉽게 다운로드 받아서 사용할 수 있다. 

```
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels = 2)
```

## 데이터셋 클래스 구현 <br>
csv 파일에서 데이터를 읽어와서 token으로 변환하는 Dataset class와 Data collator 함수를 구현하였다. 밑에서 설명할 DataLoader에서 데이터를 가져올때, Dataset class의 getitem 함수를 호출한다. Dataset class의 getitem을 통해 가져온 각 index의 데이터는 data collator 함수를 통하여 모델 인풋에 맞게 변형된다. 두 sequence를 seq token으로 구분지어 인풋을 구성하려면 tokenizer(text_a, text_b, ...) 와 같이 두 sequence를 넣어주면 된다.

```
from transformers import AutoTokenizer
from torch.utils.data import Dataset

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

class My_Dataset(Dataset):
    def __init__(self, code1, code2, similar):
        self.code1 = code1
        self.code2 = code2
        self.label = similar

    def __len__(self):
        return len(self.code1)
    
    def __getitem__(self, idx):
        code1 = self.code1[idx]
        code2 = self.code2[idx]
        label = self.label[idx]

        return code1, code2, label

def data_collator(data):
    code1 = [data_i[0] for data_i in data]
    code2 = [data_i[1] for data_i in data]
    label = [data_i[2] for data_i in data]
    
    encoded_inputs = tokenizer(code1, code2, padding=True, truncation=True, return_tensors="pt", add_special_tokens=True)
    labels = torch.tensor(label)
    
    encoded_inputs['labels'] = labels

    return encoded_inputs
```

위에서 정의한 Dataset class로 학습 및 평가 데이터의 dataset class 객체를 생성하고, dataloader를 정의하였다. dataloader 객체는 학습에 사용될 batch를 가져올때 사용된다. batch_size는 각 batch(batch는 mini-batch라고도 불린다)를 몇 개의 데이터로 구성하는지를 의미한다. shuffle은 데이터를 가져오는 순서를 섞는지 여부를 의미하며, train_loader의 경우 True로 설정하였기 때문에 각 batch에 포함되는 데이터는 실제 학습 데이터에서 랜덤한 순서로 index를 불러와 구성된다. collate_fn는 getitem 함수에서 각 index에 해당하는 데이터를 가져올 때 batch로 병합하는 함수를 설정하는 곳이다. 만약 위에서 Data_collater 함수를 따로 정의하지 않고 default 값으로 설정하여 모델을 학습하고 싶다면 Dataset class에서 getitem 함수에서 data collator에서 모델의 인풋에 맞게 데이터를 변형하는 과정을 거쳐서 값을 리턴해야한다.

```
from torch.utils.data import DataLoader

train_dataset = My_Dataset(train_code1, train_code2, train_similar)
test_dataset = My_Dataset(test_code1, test_code2, test_similar)

train_loader = DataLoader(train_dataset, batch_size = 8, shuffle=True, collate_fn=data_collator)
test_loader = DataLoader(test_dataset, batch_size = 8, shuffle=False, collate_fn=data_collator)
```

## 모델 학습 <br>
모델 학습에 필요한 optimizer와 scheduler를 정의하였다. optimizer의 경우 사전학습 언어모델을 파인튜닝할때 주로 사용하는 AdamW를 사용하였으며, scheduler 역시 사전학습 언어모델의 파인튜닝시 가장 기본적인 Liear scheduler를 사용하였다. scheduler는 학습되는 모델이 더 잘 학습될 수 있도록 learning rate의 scale을 키우거나 줄이는 역할을 한다.

```
from transformers import get_linear_schedule_with_warmup, AdamW

optimizer = AdamW(model.parameters(), lr = 2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = len(train_loader) * 4) # 4 epoch
```

아래 코드는 실제 모델이 학습되는 부분이다. nn.utils.clip_grad_norm_은 모델을 구성하는 각 파라미터의 기울기가 너무 극단적인 값을 가지지 않도록 제한하는 코드이다. 이를 통하여 더 안정적인 모델 학습이 가능하다.

```
import torch.nn as nn
from tqdm import tqdm

model = model.to(device)
model.train()

for epoch_i in range(4):       
  avg_loss = 0.0

  for batch in tqdm(train_loader):
    batch = {k:v.to(device) for k, v in batch.items()}

    optimizer.zero_grad() # 기울기 초기화

    outputs = model(**batch) # 모델 forward

    # outputs에는 logit, loss, hidden state, attention score 등의 값이 반환됨
    loss = outputs.loss

    avg_loss += loss.item() # loss 값만 누적

    loss.backward() # loss를 통해서 기울기 계산

    # 기울기 Clipping
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step() # 모델 업데이트

    scheduler.step() # scheduler 업데이트

  print(avg_loss/len(train_loader))
```


## 모델 평가 <br>
학습된 모델이 정말 잘 동작하는지 확인하기 위하여 위에서 따로 분리한 test 데이터에 대해서 정확도를 평가하였다. argmax 함수를 활용하여 모델의 출력 값인 2차원 벡터에서 더 큰 값을 가지는 차원으로 예측하도록 모델을 구현하였다. 만일 모델이 입력으로 들어온 두 코드가 유사하지 않다고 판단하는 경우(반환한 2차원 벡터에서 index 0값이 더 큰 경우) argmax 함수는 0이라는 값을 반환할 것이다. 실제 test 데이터의 정답과 모델이 예측한 값을 비교하였으며, 평가 지표로 정확도를 사용하였다.

```
model.eval()

correct = 0.0
n = 0

with torch.no_grad():
  for batch in tqdm(test_loader): 
    batch = {k:v.to(device) for k, v in batch.items()}

    outputs = model(**batch) # 모델 forward
    # outputs.loss 값 = None
    
    logits = np.array(outputs.logits.cpu())
    preds = np.argmax(logits, axis=1)

    labels = batch["labels"].cpu()
    labels = np.array(labels)

    correct += np.sum(preds == labels)
    n += len(labels)

print()
print(correct/n)
```

## 모델 개선 방안 <br>
별도의 전처리 과정이나 복잡한 모델 구조를 사용하지 않았음에도 불구하고 약 97%의 정확도를 얻을 수 있었다. 주석을 제거하거나 변수나 함수, 클래스 이름 등을 추가적으로 처리하거나, 더 복잡한 구조의 모델을 사용한다면 더 높은 성능의 모델을 학습할 수 있을 것이다.

## 구현 환경 <br>
NVIDIA RTX 3080(10GB)에서 실험을 진행하였으머 1 에폭 학습에 약 7분 정도 시간이 소요되었다. 모델 학습에 8GB의 GPU 메모리가 필요한데, 만약 GPU 메모리가 부족한 경우 batch size를 줄이거나 tokenizer에서 max sequence length 제한을 두면 더 낮은 메모리로도 학습이 가능하다.