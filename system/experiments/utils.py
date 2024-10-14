import html
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import yaml


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

def get_parent_path(level):
    path = Path(__file__)
    for i in range(level):
        path = path.parent
    return path

config = load_config(level=2)
main_path = get_parent_path(level=2)
sep_token = config['model']['sep_token']
input_token = config['model']['input_token']

def rename_col(df):
    translations = [['title','Tiêu đề'], ['body','Mô tả'], ['location','địa chỉ'], ['phone','Số điện thoại liên lạc'], ['contact_name','Tên liên lạc']
                    , ['job_type', 'Loại hình công việc'], ['education', 'Học vấn tối thiểu'], ['experience', 'Kinh nghiệm'], 
                     ['certification', 'Chứng chỉ, kỹ năng'], ['benefit', 'Các quyền lợi khác'], ['company_name', 'Tên công ty'], ['vacancy', 'Số lượng tuyển dụng']
                    , ['salary_type', 'Hình thức trả lương'], ['year_of_birth', 'Năm sinh'], ['gender', 'Giới tính'], ['age', 'Độ tuổi'], 
                    ['min_age', 'Độ tuổi tối thiểu'], ['max_age', 'Độ tuổi tối đa'], ['min_salary', 'Lương tối thiểu'], ['max_salary', 'Lương tối đa']]
    
    for i in range(len(translations)):
        try:
            df.rename(columns={translations[i][0]: translations[i][1]}, inplace=True)
        except:
            pass
    return df

def combine_input(x):
  inp = input_token
  for i in x:
      try:
        temp = str(list(x[i])[0]).replace('<br />', " ")
      except:
        temp = str(list(x[i])[0])
      inp = "{} {}: {} {} ".format(inp, i, temp, sep_token)
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

def get_input(df):
    return remove_html_tags(combine_input(df))

def format_data(X):
    title = ['title', 'job_type']
    body = ['body', 'education', 'experience', 'certification', 'benefit',]
    company = ['company_name', 'location', 'phone', 'contact_name']
    other = ['vacancy', 'total_image', 'salary_type', 'min_salary', 
            'max_salary', 'gender', 'year_of_birth' ,'age', 'min_age' , 'max_age']
    
    # Get and reposition columns
    title_df = X.loc[:,[col for col in title if col in X.columns]]
    body_df =  X.loc[:,[col for col in body if col in X.columns]]
    company_df =  X.loc[:,[col for col in company if col in X.columns]]
    other_df =  X.loc[:,[col for col in other if col in X.columns]]

    full = pd.concat([title_df, body_df, company_df, other_df], axis = 1)

    full = rename_col(full)
    return full