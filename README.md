# Speaker sever

## Table of Contents
   * [Speaker sever](#speaker-sever)
      * [method](#method)
      * [speaker code &amp; environment](#speaker-code--environment)
      * [Speaker work flow](#speaker-work-flow)
      * [speaker feature enrollment system](#speaker-feature-enrollment-system)
            * [Enroll morning conference speakers](#enroll-morning-conference-speakers)
         * [speaker enroll flow](#speaker-enroll-flow)
      * [speaker recognize API](#speaker-recognize-api)
         * [speaker server](#speaker-server)
         * [speaker server APIs](#speaker-server-apis)
      * [speaker server](#speaker-server-1)
         * [speaker server container 建立](#speaker-server-container-建立)

## method

- speaker recognition [:page_facing_up:](https://arxiv.org/pdf/1902.10107.pdf)
- GhostVLAD [:page_facing_up:](https://arxiv.org/pdf/1810.09951.pdf)
- NetVLAD [:page_facing_up:](https://arxiv.org/pdf/1511.07247.pdf)
- hackMD[:page_facing_up:](https://hackmd.io/IGa-C0_mRXqd_J6GsUCulQ)

## speaker code & environment
- python : 3.6 (:arrow_up:) 
- tensorflow : 1.14
    tf2.0其實也可以，但需要修改toolkit的funciton，因為有一些不支援
- keras : 2.3.1
    跟tf一樣
其他可參考requirement.txt

## Speaker work flow

![](https://i.imgur.com/NLuDJ9k.png)
1. enroll speaker feature
2. recognize

## speaker feature enrollment system
語者辨識需要事先註冊待辨識的人員，當進行語者辨識時會將預辨識的語音特徵向量與事先註冊好的資料進行比對，找出最相近的語者，故需要有一個註冊系統。


**註冊資料架構**
```
📁 speaker_data/
├─📁 徐泰志/
│ ├─📄 record_19.wav
│ ├─📄 record_1.wav
│ └─📄 record_17.wav

├─📁 張雅淩/
│ ├─📄 record_19.wav
│ ├─📄 record_1.wav
│ └─📄 record_17.wav
└─📁 林柏辰/
  ├─📄 record_19.wav
  ├─📄 record_1.wav
  └─📄 record_17.wav
        .
        .
        .
.
.
.
```

**input args**
data_path : 要註冊的語音資料庫
```data_path = './speaker_data'```
save_embeds_path : 生成後要存放的位置與檔名
```save_embeds_path = './speaker_feats/ai.npz'```

#### Enroll morning conference speakers
1. load morning_conference audio excel
2. get speaker's name and wave path in morning conference excel
3. output : 
    ![](https://i.imgur.com/BmURSij.png)

code : load_xls_and_save_dict.py
```python=
data_path = './server_1223/Speaker_sever/morning_conference/audio_train_30 (副本).xlsx'
# 開啟Excel檔案
xlsdata = xlrd.open_workbook(data_path, encoding_override='utf-8')
# 獲取Excel中所有的sheet
tableList = xlsdata.sheet_names()
wave_dict = {}
speakers = []
wave_path_list = []

for tablenum in range(len(tableList[:-2])):
    table = xlsdata.sheet_by_index(tablenum)
    wave_name = table.col_values(0)[1:]
    speakers += table.col_values(5)[1:]
    for num in range(len(wave_name)):
        wave_path = "./morning_conference/" + tableList[tablenum] + "/wav/" + wave_name[num] + ".wav"
        wave_path_list.append(wave_path)
        
speaker = list(set(speakers))  #消除重複人名
for name in speaker:
    path_list = []
    #找出消除重複人名後的名字(speaker)，在所有名字(speakers)中的index
    wave_path_index = [i for i,x in enumerate(speakers) if x==name]
    for index in wave_path_index:
        path_list.append(wave_path_list[index])
        wave_dict[name] = path_list

# save dict to json
with open('morning_speaker_dcit.json', 'w', encoding='utf-8') as fp:
    # ensure_ascii -> True:中文將儲存為ascii碼 ; False:可儲存中文
    json.dump(wave_dict, fp, ensure_ascii=False, indent=4)

```

### speaker enroll flow
1. load wavs path
2. generate speaker feats
3. save feats to npz (default not save)

code : speaker_enroll_v1.py
程式重點部份我已經打好註解
```python=
def flow(self, data_path, save_embeds_path=None):
"""
speaker embeds enroll flow
1. load wavs path
2. generate speaker feats
3. save feats to npz (default not save)
"""
self.data_path = data_path
# load wav path
self.wav_paths = self.load_wav_paths(self.data_path)

# generate speaker feats
self.speaker_embeds = self.generate_speaker_embeds()

# save feats to npz
if save_embeds_path:
    np.savez(save_embeds_path, label=np.array(list(self.wav_paths.keys())), train_feats=self.speaker_embeds)

print(f"save speaker embeds : {save_embeds_path}")
```


## speaker recognize API 
### speaker server
目前已經註冊的speaker人員有三類：
1. all
2. ai  (人工智慧組44人）--> feature_muti_ai_Feb20.npz
3. moring  (晨會長官 12天）--> feature_morning_muti1-12.npz
speaker recognize server只能辨識這三類中的人員，若要新增新的default speaker mode，則須修改speaker server讀取的資料庫。

### speaker server APIs
**InitSpeakerFeats**
```python
def test_InitSpeakerFeats():
    # InitSpeakerFeats : init speaker server to default speaker feats list
    # default mode : all
    resp = requests.post(url + "InitSpeakerFeats")
    resp_data = json.loads(resp.text)
    print(resp_data)
```

**speaker**
語者辨識，會依照當前server設定的speaker list做辨識，找出一個與此語音最相近的語者
```python
def test_speaker(file):
    # predict speaker
    ## load wav
    sr = 16000
    s = time.time()
    wav, sr = librosa.load(file, sr=sr)
    
    if not isinstance(wav, str):
        data = {"speaker":wav.tolist()}
        resp = requests.post(url + "speaker",json=data)
        e = time.time()
        print(e-s)
        resp_data = json.loads(resp.text)
        print(resp_data)
        
    return resp_data
```

**GetSpeakerList**
取得現在server中的所有資訊，包含了speaker label, feats, mode
```python=
def test_GetSpeakerList():
    # GetSpeakerList : if want to know current speaker mode and label
    # resp_data -> {label", "feats", "feats_mode"}
    resp = requests.post(url + "GetSpeakerList")
    resp_data = json.loads(resp.text)
    print("feats_mode : " + resp_data["feats_mode"])
    print(resp_data["label"])
    
    return resp_data
```

**ModifyFeatsMode**
修改speaker mode 
現在有三個mode:
1. all
2. ai  (人工智慧組44人）
3. moring  (晨會長官 12天）
```python=
def test_ModifyFeatsMode(data):
    # ModifyFeatsMode : modify speaker mode
    '''
    post -> data : {"feats_mode": mode}
        mode = "all", "moring", "ai"
    resp_data -> {label", "feats_mode"}
    '''
    resp = requests.post(url + "ModifyFeatsMode", json=data)
    resp_data = json.loads(resp.text)
    print("feats_mode : " + resp_data["feats_mode"])
    print(resp_data["label"])
    
    return resp_data
```

**ModifyFeatsCustom**
自訂義speaker feats and label
輸入label和對應的feats，則可針對此名單做辨識
```python=
def test_ModifyFeatsCustom(data):
    # ModifyFeatsCustom : modify custom speaker feas and label
    '''
    post -> data : {"label": custom_label, "feats": custom_feats}
    resp_data -> {label", "feats_mode"}
    '''
    resp = requests.post(url + "ModifyFeatsCustom", json=data)
    resp_data = json.loads(resp.text)
    print("feats_mode : " + resp_data["feats_mode"])
    print(resp_data["label"])
    
    return resp_data
```

## speaker server
run 裡面的speaker_server.py開始speaker server就好了

### speaker server container 建立
建立與註冊完[morning conference speakers feats](https://hackmd.io/nNpyEUELT3OP_rLIXhDPPQ?view#Enroll-morning-conference-speakers)後，在更新speaker_server_always_gpu時因程式改錯，導致外部網路連不上此container，因此需重新建立speaker server container...
```
docker run --runtime nvidia --name speaker_server -it -p 6001:100 -e LANG=C.UTF-8 speaker:v1.0-109 /bin/bash
```
將container speaker_server commit成一image為speaker:v1.1-1100225
```
docker commit speaker_server speaker:v1.1-110225
```
最後基於speaker:v1.1-1100225的image建立可一直重起的server container speaker_server_always_gpu
```
docker run --runtime nvidia --name speaker_server_always_gpu -it --restart=always -p 6001:100 -e LANG=C.UTF-8 speaker:v1.1-1100225 /bin/sh /server.sh
```
