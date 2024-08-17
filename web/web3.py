import torch
import torch.nn as nn
from torch.nn.functional import softmax
from pywebio.platform.tornado import start_server
from pywebio import STATIC_PATH
from pywebio.input import file_upload
from tornado.ioloop import IOLoop
from pywebio.input import *
from pywebio.output import *
from ipywidgets import Button, Output, VBox
from IPython.display import display, Javascript, HTML
import os
import io
import pandas as pd

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x

ls=('ARHGAP21_mut_important', 'ARHGAP21_cnv_del', 'ARHGAP21_cnv_amp',
       'RBBP5_mut_important', 'RBBP5_cnv_del', 'RBBP5_cnv_amp',
       'PPP1CA_mut_important', 'PPP1CA_cnv_del', 'PPP1CA_cnv_amp',
       'CTSE_mut_important', 'CTSE_cnv_del', 'CTSE_cnv_amp',
       'UBE2W_mut_important', 'UBE2W_cnv_del', 'UBE2W_cnv_amp',
       'AR_mut_important', 'AR_cnv_del', 'AR_cnv_amp', 'MUC4_mut_important',
       'MUC4_cnv_del', 'MUC4_cnv_amp', 'COL1A2_mut_important',
       'COL1A2_cnv_del', 'COL1A2_cnv_amp', 'PKP1_mut_important',
       'PKP1_cnv_del', 'PKP1_cnv_amp', 'CD55_mut_important', 'CD55_cnv_del',
       'CD55_cnv_amp', 'MUC16_mut_important', 'MUC16_cnv_del', 'MUC16_cnv_amp',
       'PTEN_mut_important', 'PTEN_cnv_del', 'PTEN_cnv_amp',
       'TP53_mut_important', 'TP53_cnv_del', 'TP53_cnv_amp',
       'FGF2_mut_important', 'FGF2_cnv_del', 'FGF2_cnv_amp',
       'ARHGEF9_mut_important', 'ARHGEF9_cnv_del', 'ARHGEF9_cnv_amp',
       'RB1_mut_important', 'RB1_cnv_del', 'RB1_cnv_amp',
       'ASB12_mut_important', 'ASB12_cnv_del', 'ASB12_cnv_amp',
       'RNF213_mut_important', 'RNF213_cnv_del', 'RNF213_cnv_amp')
# 创建模型实例并加载保存的参数
input_size = len(ls)
hidden_size = 100
output_size = 2
# MLP实例化
model = MLP(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('MLP.pth'))
model.eval()

def mlp_web_app():

    file_mut = file_upload("Select mutation file")  # 用户选择文件
    content = file_mut['content']
    mut = io.BytesIO(content)
    file_cnv = file_upload("Select cnv file")  # 用户选择文件
    content = file_cnv['content']
    cnv = io.BytesIO(content)

    df_mut = pd.read_csv(mut, sep='\t', low_memory=False, skiprows=0)
    id_col = 'Tumor_Sample_Barcode'
    include = ['Translation_Start_Site', 'Nonstop_Mutation', 'Splice_Site', 'Frame_Shift_Del', 'Nonsense_Mutation',
               'Missense_Mutation', 'Frame_Shift_Ins', ' In_Frame_Del', 'In_Frame_Ins']
    df_mut = df_mut[df_mut['Variant_Classification'].isin(include)].copy()
    mut_table = pd.pivot_table(data=df_mut, index=id_col, columns='Hugo_Symbol', values='Variant_Classification',
                              aggfunc='count')
    df_table = mut_table.fillna(0)

    df_cnv = pd.read_csv(cnv, sep='\t', low_memory=False, skiprows=0, index_col=0)
    df_cnv = df_cnv.T
    df = df_cnv.fillna(0.)
    df = df.drop('Entrez_Gene_Id', axis=0)

    # 转换突变表格
    mutation_table = df_table.apply(lambda x: x >= 1).astype(int)
    # 转换拷贝数表格
    copy_number_table = df.applymap(lambda x: 1 if x == 2 else (-1 if x == -2 else 0))

    amp_table = copy_number_table.copy()
    amp_table[amp_table < 0] = 0

    # 创建del表格，大于0的数替换为0，-1替换成1
    del_table = copy_number_table.copy()
    del_table[del_table > 0] = 0
    del_table[del_table == -1] = 1

    # Extract gene columns from each table
    cols_list = [set(mutation_table.columns), set(amp_table.columns), set(del_table.columns)]
    cols = set.union(*cols_list)
    all_cols = list(cols)

    all_cols_df = pd.DataFrame(index=all_cols)
    df_list = []
    for df, data_type in zip([mutation_table, amp_table, del_table], ['mut_important', 'cnv_amp', 'cnv_del']):
        df = df.copy()
        df = df[df.columns.intersection(cols)]
        df = df.T.join(all_cols_df, how='right')
        df = df.T
        df = df.fillna(0)
        df_list.append(df)
    all_data = pd.concat(df_list, keys=['mut_important', 'cnv_amp', 'cnv_del'], join='inner', axis=1)
    # put genes on the first level and then the data type
    data = all_data.swaplevel(i=0, j=1, axis=1)
    data = data.loc[:, [col for col in data.columns if col[1] in ["mut_important", "cnv_del", "cnv_amp"]]]
    data.columns = data.columns.get_level_values(0) + "_" + data.columns.get_level_values(1)
    # data_filter = data.loc[:, ls]
    data_filter = data.copy()
    for col_name in ls:
        if col_name not in data.columns:
            data_filter[col_name] = 0
    data_filter = data_filter.loc[:, ls]
    print(data_filter.head(3))

    put_markdown(r""" # Prostate Cancer Metastasis Prediction""").show()
    # put_image('https://www.python.org/static/img/python-logo.png').show()
    image_path = 'D:/zhuomian/web/figure2.jpg'

    put_image(open(image_path, 'rb').read())
    put_markdown(r""" # results
     """).show()
    # 推理
    with torch.no_grad():
        model.eval()
        input_tensor = torch.tensor(data_filter.values, dtype=torch.float32)
        outputs = model(input_tensor)
        probabilities = softmax(outputs, dim=1)
        probs = probabilities.numpy()
        prob1 = probs[:, 0].round(2)
        prob2 = probs[:, 1].round(2)


    # 输出表格
    table_data = []
    for index, prediction1, prediction2 in zip(data_filter.index, prob1, prob2):
        table_data.append([index] + [prediction1] + [prediction2])

    table = [
        ['Sample', 'Tumor stability', 'Tumor deterioration'],
        *table_data
    ]
    put_table(table)

    # 显示表格和筛选控件
    while True:
        # 获取筛选条件
        keyword = input("Please enter a keyword：", type=TEXT)

        # 表格1
        filtered_data = [
            row for row in table
            if (not keyword or any(keyword.lower() in str(cell).lower() for cell in row))
        ]
        print(filtered_data)
        select_table = [
            ['Sample', 'Tumor stability', 'Tumor deterioration'],
            *filtered_data
        ]
        # 表格2
        filtered_data1 = data_filter.loc[[keyword], :]
        ls1 = ('AR_cnv_del', 'AR_cnv_amp', 'AR_mut_important'
                , 'PTEN_cnv_del', 'PTEN_cnv_amp', 'PTEN_mut_important'
                , 'TP53_cnv_del', 'TP53_cnv_amp', 'TP53_mut_important'
                , 'MUC4_cnv_del', 'MUC4_cnv_amp', 'MUC4_mut_important'
                , 'RB1_cnv_del', 'RB1_cnv_amp', 'RB1_mut_important'
                , 'RBBP5_cnv_del', 'RBBP5_cnv_amp', 'RBBP5_mut_important'
                , 'CD55_cnv_del', 'CD55_cnv_amp', 'CD55_mut_important'
                , 'ARHGEF9_cnv_del', 'ARHGEF9_cnv_amp', 'ARHGEF9_mut_important'
                , 'ASB12_cnv_del', 'ASB12_cnv_amp', 'ASB12_mut_important'
                , 'CTSE_cnv_del', 'CTSE_cnv_amp', 'CTSE_mut_important'
                , 'FGF2_cnv_del', 'FGF2_cnv_amp', 'FGF2_mut_important'
                , 'RNF213_cnv_del', 'RNF213_cnv_amp', 'RNF213_mut_important'
                , 'MUC16_cnv_del', 'MUC16_cnv_amp', 'MUC16_mut_important'
                , 'UBE2W_cnv_del', 'UBE2W_cnv_amp', 'UBE2W_mut_important'
                , 'PKP1_cnv_del', 'PKP1_cnv_amp', 'PKP1_mut_important'
                , 'PPP1CA_cnv_del', 'PPP1CA_cnv_amp', 'PPP1CA_mut_important'
                , 'ARHGAP21_cnv_del', 'ARHGAP21_cnv_amp', 'ARHGAP21_mut_important'
                , 'COL1A2_cnv_del', 'COL1A2_cnv_amp', 'COL1A2_mut_important')
        print(filtered_data1)
        filtered_data1 = filtered_data1.loc[:, ls1]
        print(filtered_data1)
        select_table1 = [filtered_data1.columns.tolist()] + filtered_data1.values.tolist()
        print(select_table1)


        with use_scope('scope1', clear=True):
            put_table(select_table)
            put_table(select_table1)



if __name__ == '__main__':
    start_server(mlp_web_app, port=800)
    IOLoop.current().start()

