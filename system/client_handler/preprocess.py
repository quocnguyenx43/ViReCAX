import pandas as pd
import numpy as np

import requests
import json

from bs4 import BeautifulSoup
import html
# from tqdm.auto import tqdm

import re
import yaml
from pathlib import Path

def load_config(level, path=''):
    config_folder = Path(__file__)
    if path == '':
        for i in range(level):
            config_folder = config_folder.parent
    else:
        config_folder = Path(__file__).parent + path

    with open(f"{config_folder}\config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config(level = 2)
sep_token = config['model']['sep_token']
input_token = config['model']['input_token']

API_DETAIL = 'https://muaban.net/listing/v1/classifieds/'
API_USER = 'https://muaban.net/listing/v1/users/'

def combine_input(x):
  inp = input_token
  for i in x:
      temp = str(x[i]).replace('<br />', " ")
      inp = "{} {}: {} {} ".format(inp, temp, x[i][0], sep_token)
  return inp

def remove_html_tags(text):
    # Sử dụng BeautifulSoup để phân tích cú pháp HTML
    text = text.replace('\t', " ")
    text = text.replace('\n', " ")
    soup = BeautifulSoup(text, "html.parser")

    # Xóa các tag HTML trong đoạn văn bản
    cleaned_text = soup.get_text()

    # Xóa các ký tự thuộc HTML (ví dụ: &nbsp;)

    cleaned_text = html.unescape(cleaned_text)
    return cleaned_text

def crawl_detail(w_api, id):
    detail = []
    t = requests.get("{}{}/detail".format(w_api, id)).text
    t = json.loads(t)
    detail.append(t)
    detail = pd.DataFrame.from_dict(detail)
    return detail

def crawl_user(w_api, u_id, id):
    users = []
    res = requests.get(w_api + str(int(u_id)) + '/profile').text
    res = json.loads(res)
    res['ad_id'] = id
    users.append(res)
        
    users = pd.DataFrame.from_dict(users)
    return users

def process_details(details_df):
    # print('Processing details_df ...')
    # Processing phone number
    # print('1. Processing phone number')
    details_df.phone = details_df.phone.astype(str).map(lambda x: '0' + x).map(lambda x: x.replace('0nan', '')).map(lambda x: x.replace('.0', ''))

    # Removing all recruiments have no `user_id`
    # print('2. Removing all recruiments have no `user_id``')
    details_df = details_df[details_df.user_id.notnull()].reset_index(drop=True)
    details_df = details_df[details_df.user_id != 0].reset_index(drop=True)

    # Convert params to right format
    def convert_parameters(data):
        # print('Converting params to right format ...')
        # data = ast.literal_eval(data) # for data is str
        converted_data = {}
        for item in data:
            label = item['label']
            value = item['value']
            converted_data[label] = value
        # print('Converting params to right format is successfully...')
        return converted_data

    # Convert salary to right format
    def convert_salary(df):
        # print('Converting salary to right format ...')
        t = df[['price_display']]
        pattern = r'\d{1,3}(?:\.\d{3})*'
        min_sal = []
        max_sal = []

        for i, r in t.iterrows():
            sal = r.price_display
            min_s = np.nan  # Khởi tạo min_s và max_s trước khi sử dụng
            max_s = np.nan
            if sal == 'Thỏa thuận':
                min_s = np.nan
                max_s = np.nan
            else:
                matches = re.findall(pattern, sal)
                if len(matches) == 2:
                    min_s = matches[0]
                    max_s = matches[1]
                elif 'Từ' in sal:
                    min_s = matches[0]
                    max_s = np.nan
                elif 'Đến' in sal:
                    max_s = matches[0]
                    min_s = np.nan

            min_sal.append(min_s)
            max_sal.append(max_s)
        
        df['min_salary'] = min_sal
        df['max_salary'] = max_sal

        # print('Converting salary to right format is successfully...')
        return df

    # Convert params
    # print('3. Converting params to right format')
    params = pd.json_normalize(details_df.parameters.map(convert_parameters))
    details_df = pd.concat([details_df, params], axis=1)

    # Convert salary
    # print('4. Converting salary to right format')
    details_df = convert_salary(details_df)

    return details_df

def process_users(user):
        # print('Processing users ...')

        # Phone
        user.phone = user.phone.astype(str).map(lambda x: '0' + x).map(lambda x: x.replace('0nan', '')).map(lambda x: x.replace('.0', ''))
        
        # Dropping all used cols
        dropped_cols = [
            'microsite', 'total',
            'post_hidden_phone', 'post_hidden_microsite',
            'reputation', 'image_url'
        ]
        for col in dropped_cols:
            try:
                user.drop(col, axis=1, inplace=True)
            except:
                continue

        # Changing the col names
        try:
            user.columns = [
                'ID người đăng', 'Tên người đăng', 'Số điện thoại người đăng', 'URL người đăng', 'u.created_date', 'u.ad_id'
            ]
        except:
            print('Cannot change column names!')
        user['u.ad_id'].astype(int)
        
        # print('Processing users is successfully!')
        return user

def rename_col(df):
    translations = [['title','Tiêu đề'], ['body','Mô tả'], ['location','địa chỉ'], ['phone','Số điện thoại liên lạc'], ['contact_namme','Tên liên lạc'], 
     ['pulish_at', 'Thời gian đăng bài'], ['submission_expired', 'Thời gian hết hạn'], ['is_anonymous','Là người lạ'], ['is_recruiters','Là người tuyển dụng'],
     ['total_image','Số lượng ảnh']]
    
    for i in range(len(translations)):
        try:
            df.rename(columns={translations[i][0]: translations[i][1]}, inplace=True)
        except:
            pass
    return df


def crawl_full(d_api, u_api, id):

    title = ['title', 'Ngành nghề']
    body = ['body', 'Học vấn tối thiểu', 'Kinh nghiệm', 'Chứng chỉ, kỹ năng', 'Các quyền lợi khác',]
    company = ['Tên công ty', 'location', 'phone', 'contact_name']
    poster = ['ID người đăng', 'Tên người đăng', 'Số điện thoại người đăng', 'URL người đăng', 'publish_at', 'submisson_expired','is_anonymous', 'is_recruiters']
    other = ['id', 'url', 'Số lượng tuyển dụng', 'total_image', 'Loại hình công việc', 'Hình thức trả lương', 'Lương tối thiểu', 
             'Lương tối đa', 'Giới tính', 'Độ tuổi']
    
    detail = crawl_detail(d_api, id)
    d = process_details(detail)

    user = crawl_user(u_api, str(d['user_id'][0]), str(id))
    u = process_users(user)

    full = pd.concat([d, u], axis =1 )

    full_col = full.columns

    # Get and reposition columns
    title_df = full.loc[:,[col for col in title if col in full_col]]
    body_df =  full.loc[:,[col for col in body if col in full_col]]
    company_df =  full.loc[:,[col for col in company if col in full_col]]
    poster_df =  full.loc[:,[col for col in poster if col in full_col]]
    other_df =  full.loc[:,[col for col in other if col in full_col]]

    full = pd.concat([title_df, body_df, company_df, poster_df, other_df], axis = 1)
    
    full = rename_col(full)

    return full
    # return d, u

def get_input(df):
    return remove_html_tags(combine_input(df))



    