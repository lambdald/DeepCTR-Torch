# -*- coding: utf-8 -*-
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *
from sklearn.utils import shuffle
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter   


if __name__ == "__main__":
    data = pd.read_csv('./data/output.csv')
    data = shuffle(data)



    ic_vector = data['main_vec_comm_i2v_ic_item_vector'].fillna('').map(lambda x: np.array(list(map(float, x.strip('\'[]').split()))))

    ic_vector_list = []
    for i in range(len(ic_vector)):
        ic_vector_list.append(ic_vector[i])

    ic_vector = np.array(ic_vector_list)

    ic_vector_cols = [f'main_vec_comm_i2v_ic_item_vector_{i:02}' for i in range(ic_vector.shape[1])]

    ic_vector = pd.DataFrame(ic_vector, columns=ic_vector_cols)
    data = pd.concat([data,ic_vector], axis=1)




    interest_vector = data['vec_comm_i2v_b_ln_buyer_long_interest'].fillna('').map(lambda x: np.array(list(map(float, x.strip('\'[]').split()))))

    ic_vector_list = []
    for i in range(len(interest_vector)):
        ic_vector_list.append(interest_vector[i])

    interest_vector = np.array(ic_vector_list)

    interest_cols = [f'vec_comm_i2v_b_ln_buyer_long_interest_{i:02}' for i in range(ic_vector.shape[1])]

    interest_vector = pd.DataFrame(interest_vector, columns=interest_cols)
    data = pd.concat([data,interest_vector], axis=1)






    data['comm_b_basic_buyer_phone_os_rank'] = data['comm_b_basic_buyer_phone_os'].rank(ascending=0, method='dense')


    sparse_features = ['comm_b_basic_buyer_age', 'main_comm_ic_behavior_buyer_30d', 'vec_comm_i2v_b_ln_buyer_long_interest', 'comm_b_basic_buyer_phone_os_rank'] + interest_cols
    dense_features = ['main_comm_ic_basic_pic_score_ai', 'main_comm_ic_basic_prod_score'] + ic_vector_cols


    data[sparse_features] = data[sparse_features].fillna(-1, )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['click']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=4)
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   task='binary',
                   l2_reg_embedding=1e-5, device=device)

    model.compile("adagrad", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )


    writer = SummaryWriter('exp/tensorboard')


    history = model.fit(train_model_input, train[target].values, batch_size=1024, epochs=100, verbose=2,
                        validation_split=0.2, tb_writer=writer)
    
    model_path = Path('model/deepfm.pth')
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_path)

    

    pred_ans = model.predict(test_model_input, 256)
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
