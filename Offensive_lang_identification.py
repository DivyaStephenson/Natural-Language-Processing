from transformers import BertTokenizer, BertModel
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pandas as pd
import torch
import transformers as ppb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import f1_score, confusion_matrix,classification_report
from xgboost import XGBClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import preprocessor as p
from sklearn.svm import SVC



class MBert_svm:

    def __init__(self):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.model = BertModel.from_pretrained("bert-base-multilingual-cased")

    def read_from_excel(self,path):

        script_dir =  os.path.dirname(os.path.abspath("__file__"))   
        df_all = pd.read_csv(os.path.join(script_dir,path), sep='\t', header=None)
        return df_all

    def encode_dataset(self, df):
        print('Tokn',self.tokenizer)
        tokenized = df['Tweet'].apply((lambda x: self.tokenizer.encode(x, add_special_tokens=True)))
        print(tokenized)
        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)
        padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values]) #padding with zeroes 
        print(padded)
        attention_mask = np.where(padded != 0, 1, 0)
        attention_mask.shape
        input_ids = torch.tensor(padded) #.to(torch.int64)
        attention_mask = torch.tensor(attention_mask)
        with torch.no_grad(): 
            last_hidden_states = self.model(input_ids, attention_mask=attention_mask)
        return last_hidden_states

    def dataet_split(self, df, last_hidden_states):
        features = last_hidden_states[0][:,0,:].numpy()
        labels = df['label']
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels)
        return train_features, test_features, train_labels, test_labels

    def Logestic_regression_model(self, train_features, train_labels):

        lr_clf = LogisticRegression(solver='lbfgs', max_iter=100)
        lr_clf.fit(train_features, train_labels)
        return lr_clf

    
    def SVM_sample(self, train_features, train_labels):

        classifier = SVC(kernel='linear', random_state=0)  
        classifier.fit(train_features, train_labels)
        return classifier

    def xboost_classifier(self, train_features, train_labels):

        # fit xboost
        model = XGBClassifier()
        model.fit(train_features, train_labels)
        return model

    def random_forest(self, train_features, train_labels):

        clf=RandomForestClassifier(n_estimators=100)
        clf.fit(train_features, train_labels)
        return clf

    def Llinear_discriminant_analysis(self, train_features, train_labels):
        clf = LinearDiscriminantAnalysis()
        clf.fit(train_features, train_labels)
        return clf

from googletrans import Translator, constants
from better_profanity import profanity
translator = Translator()

def google_trans(sentence):

    translator = Translator()
    translation = translator.translate(sentence,src="ta", dest="en")
    return translation.text

def check_profound(sent):
    return profanity.contains_profanity(sent)

import preprocessor as p

if __name__ == "__main__":
    # training the model 
    bert = MBert_svm()
    
    #Train file path
    path = ' '
    data_frame = bert.read_from_excel(path)
    data_frame['Tweet'] = [p.clean(str(i)) for i in data_frame[1].apply(str)]
    data_frame['label'] = data_frame[2]
    print(data_frame.head())
    hidden_layer = bert.encode_dataset(data_frame)
    features = hidden_layer[0][:,0,:].numpy()
    log_model = bert.Logestic_regression_model(features, data_frame['label'])
    Xboost_model = bert.xboost_classifier(features, data_frame['label'])
    svm_model = bert.SVM_sample(features, data_frame['label'])
    random_forest_model = bert.random_forest(features, data_frame['label'])
    lda_model = bert.Llinear_discriminant_analysis(features, data_frame['label'])
    final_dic = {}
    
    #model evaluation
    #Test File path
    path_evaluation=' '
    data_frame_eval = bert.read_from_excel(path)
    data_frame_eval['Tweet'] = [p.clean(str(i)) for i in data_frame_eval[1]]
    data_frame_eval['label'] =  data_frame_eval[2]


    hidden_layer_eval = bert.encode_dataset(data_frame_eval)
    features = hidden_layer_eval[0][:,0,:].numpy()
    logistic_pred = log_model.predict(features)
    print("logistic Regression",logistic_pred)
    final_dic['logistic_pred'] = logistic_pred
    xboost_pred = Xboost_model.predict(features)
    print("xboost Regression",xboost_pred)
    final_dic['xboost_pred'] = xboost_pred
    svm_pred = svm_model.predict(features)
    print("svm Regression",svm_pred)
    final_dic['svm_pred'] = svm_pred
    random_pred = random_forest_model.predict(features)
    print("random Regression",random_pred)
    final_dic['random_pred'] = random_pred
    lda_pred = lda_model.predict(features)
    print("lda Regression",lda_pred)
    final_dic['lda_pred'] = lda_pred


    before_prof = []
    after_prof = []

    for i in range(len(logistic_pred)):

        temp_val = None
        after_pref = None
        temp_list = []
        temp_list.append(xboost_pred[i])
        temp_list.append(random_pred[i])
        temp_list.append(lda_pred[i])
        print(temp_list)
        if temp_list.count('OFF') >=2:
            temp_val = 'OFF'
        if temp_list.count('NOT') >=2:
            temp_val = 'NOT'

        before_prof.append(temp_val)
        if temp_val is 'NOT':
        
            translated_sentance = google_trans(data_frame_eval['Tweet'][i])
            profound_sentence = check_profound(translated_sentance)
            if profound_sentence is True:
                after_pref = 'OFF'
            else:
                after_pref = 'NOT'
        else:
            after_pref = 'OFF'

        after_prof.append(after_pref)
    f1_before_pred = f1_score( data_frame_eval['label'], before_prof, average='weighted')
    f1_after_pred = f1_score( data_frame_eval['label'], after_prof, average='weighted')

    print("before profound  f1 score : ",f1_before_pred)

    print("After perfound  f1 score : ",f1_after_pred)

    print(classification_report(data_frame_eval['label'], after_prof, labels=['NOT','OFF']))










