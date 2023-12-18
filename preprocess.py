import datetime
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
import os 
import parmap 
import pickle 
import json  
import glob
import numpy as np
import pandas as pd 
import argparse 
from PIL import Image  

parser = argparse.ArgumentParser(description='학습 스크립트에 대한 인자 설정')
parser.add_argument('--root_path', type=str, default='/data/output_20231211', help='원본 데이터 경로')
parser.add_argument('--pretrain_scene_list_path', type=str, default='pretrain_model/pretrain_scene_list.csv',
                    help='pretrain에 사용한 데이터 목록 csv 파일 경로')


# 인자 파싱
args = parser.parse_args()
root_path = args.root_path # 학습 관련 상수 설정 
raw_image_path = f'{root_path}/원천데이터/' # 이미지 경로 
raw_scene_path = f'{root_path}/라벨링데이터/20231211_output_kg.json' # 장면 그래프 경로 
raw_qa_path = f'{root_path}/라벨링데이터/gqa_json_all_in_one.json'# 질의응답 경로
image_pickle_path = f'{root_path}/가공데이터/' # pkl로 변환된 이미지가 저장될 폴더
os.makedirs(image_pickle_path, exist_ok=True) 

pretrain_scene_list_path = args.pretrain_scene_list_path # 사전훈련된 모델 경로
num_cores = os.cpu_count() 
          
def read_json_from_file(file_path):
    """
    주어진 파일 경로에서 JSON 파일을 읽고 파이썬 객체로 반환합니다.
    :param file_path: 읽을 파일의 경로
    :return: JSON 데이터를 포함한 파이썬 객체
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_and_format_scene_graph_data(raw_scene_path):
    """
    장면 그래프 데이터를 로드하고 메인 데이터프레임을 생성합니다.
    :param raw_scene_path: 장면 그래프 JSON 파일 경로
    :return: 생성된 메인 데이터프레임
    """
    data = read_json_from_file(raw_scene_path)
    df_main = pd.DataFrame(data)
    return df_main 
 
# 질의응답 데이터프레임으로 변환
def json_to_dataframe(json_data):
    # 데이터프레임을 만들기 위한 빈 리스트
    df_list = []

    # JSON 데이터를 순회하면서 필요한 정보 추출
    for qa_data in json_data:
        scene_id = qa_data['Scene_Graph_ID']
        for qa in qa_data['QA_list']:
            df_list.append({
                'sceneId': scene_id,
                'QA_ID': qa['QA_ID'],
                'question': qa['question'],
                'answer': qa['answer'],
                'question_type': ', '.join(qa['question_type']),
                'answer_type': qa['answer_type']
            })
    # 리스트를 데이터프레임으로 변환
    return pd.DataFrame(df_list) 

df_main  = load_and_format_scene_graph_data(raw_scene_path) 

# 질의응답 JSON 파일을 로드하고 데이터프레임으로 변환한 후, 메인 데이터프레임과 병합

start = datetime.datetime.now() #셀 시작시간 측정 
json_file = read_json_from_file(raw_qa_path)
all_df = json_to_dataframe(json_file)
all_df = all_df[['sceneId','QA_ID', 'question', 'answer', 'question_type', 'answer_type']]
all_df.rename({'sceneId' :'Scene_Graph_ID'}, axis = 1, inplace = True)  
all_df = all_df[all_df['answer_type'] == 'full_answer']

# 메인 데이터프레임과 병합
all_df = pd.merge(all_df, df_main[['Scene_Graph_ID', 'Category',]]) 
all_df = all_df[['Scene_Graph_ID','Category', 'question', 'answer']] 

all_df['image_path'] = all_df['Category'].apply(lambda x:raw_image_path+x) 
all_df['image_path'] = all_df['image_path'] + '/' + all_df['Scene_Graph_ID']+ '.jpg' 
all_df['path'] = image_pickle_path + all_df['Scene_Graph_ID']+ '.jpg'
end = datetime.datetime.now()
sec = (end - start)
print('Json 로드 소요 시간 : ',str(sec)) 


# 데이터를 Train/Validation/Test로 나누는 과정
# 1. 유니크 이미지 식별
# 2. 이미지 기반 데이터 분할 : 식별된 유니크한 이미지들을 Train/Validation/Test 세트로 나누고, 이미지를 기반으로 질문/답변 데이터를 나눔 

# 전체 질의응답의 scene_graph_id 목록
unique_image_list = all_df.Scene_Graph_ID.unique() 


print('unique_image_list',unique_image_list.shape[0])


if os.path.exists('pretrain_model/pretrain_scene_list.csv'): 
    # pretrain에 사용한 scene_graph_id 목록
    pretrain_scene_list = pd.read_csv(pretrain_scene_list_path, index_col=0)
    pretrain_scene_list = pretrain_scene_list['image_id'].unique()
    print('pretrain_scene_list',pretrain_scene_list.shape[0]) 
else: 
    pretrain_scene_list = [] 
    print('pretrain에 사용한 데이터 발견 x')

# pretrain 데이터가 전체 데이터의 80% 이상인 경우    
if pretrain_scene_list.shape[0] > unique_image_list.shape[0]*0.8: 
        print('pretrain 데이터가 전체 데이터의 80% 초과')
        train_img_list = pretrain_scene_list
        val_and_test = unique_image_list[~np.isin(unique_image_list,train_img_list)]
        valid_img_list,test_img_list = train_test_split(val_and_test,test_size=0.5,random_state=22)
        print('train_img_list', train_img_list.shape[0])
        print('valid_img_list', valid_img_list.shape[0])
        print('test_img_list', test_img_list.shape[0])
# pretrain 데이터가 전체 데이터의 80% 미만인 경우
else:
    # 전체 데이터에서 pretrain 데이터를 제외한 나머지 데이터
    remaining_data = unique_image_list[~np.isin(unique_image_list, pretrain_scene_list)]
    # train 세트에 필요한 추가 데이터의 양 계산
    additional_train_len = int(len(unique_image_list) * 0.8) - len(pretrain_scene_list)
    # 추가 train 데이터 선택
    additional_train_data, remaining_after_train = train_test_split(remaining_data, train_size=additional_train_len, random_state=22)
    # 남은 데이터를 검증과 테스트 세트로 나눔
    valid_img_list, test_img_list = train_test_split(remaining_after_train, test_size=0.5, random_state=22)
    # train 세트는 pretrain 데이터와 추가 train 데이터를 합쳐서 구성
    train_img_list = np.concatenate((pretrain_scene_list, additional_train_data))

# 결과 출력
print('train_img_list', train_img_list.shape[0])
print('valid_img_list', valid_img_list.shape[0])
print('test_img_list', test_img_list.shape[0]) 

train_x = all_df[all_df['Scene_Graph_ID'].isin(train_img_list)] 
valid_x = all_df[all_df['Scene_Graph_ID'].isin(valid_img_list)] 
test_x = all_df[all_df['Scene_Graph_ID'].isin(test_img_list)] 

print('train_x.shape', train_x.shape)
print('valid_x.shape', valid_x.shape)
print('test_x.shape', test_x.shape) 

train_x.to_csv('./train.csv') 
valid_x.to_csv('./valid.csv')
test_x.to_csv('./test.csv') 

# 이미지 pkl로 저장 

def process_and_save_image(args):
    """
    이미지 경로를 입력받아, 이미지를 로드하고 결과를 .pkl 파일로 저장합니다.

    Parameters:
    - image_path (str): 처리할 이미지의 파일 경로.
    - save_path (str): 결과를 저장할 .pkl 파일의 경로.
    """ 
    try:
        image_path, save_path = args
        # 이미지 로드
        image = Image.open(image_path).convert('RGB')
        
        # 결과를 pkl 파일로 저장
        with open(save_path, 'wb') as f:
            pickle.dump(image, f) 

    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")         
        
image_list = all_df['image_path'] 
image_pkl_list = list(map(lambda x : image_pickle_path+x.split('/')[-1].replace('.jpg','.pkl'), image_list)) 

parmap.map(process_and_save_image, list(zip(image_list, image_pkl_list)), pm_pbar=True, pm_processes=num_cores)
end = datetime.datetime.now()
sec = (end - start)
print('총 이미지 개수 : ', len(image_list))
print('총 소요 시간 : ',str(sec))