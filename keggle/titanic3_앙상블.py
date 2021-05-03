# https://www.kaggle.com/kalashnimov/tps-apr-simple-ensemble

import numpy as np 
import pandas as pd

sub = pd.read_csv('C:/data/keggle/taita/sample_submission.csv')
sub1 = pd.read_csv('C:/data/keggle/taita/tps04-sub-003.csv') 
sub2 = pd.read_csv('C:/data/keggle/taita/AutoWoE_submission_combo.csv') 
sub3 = pd.read_csv('C:/data/keggle/taita/submission_pseudo_test.csv') 
sub4 = pd.read_csv('C:/data/keggle/taita/voting_submission.csv')
sub5 = pd.read_csv("C:/data/keggle/taita/answer/submission4_0429.csv")
sub6 = pd.read_csv("C:/data/keggle/taita/answer/voting_submission2.csv")

res = (4*sub1['Survived'] + sub2['Survived'] + sub3['Survived'] + 4*sub4['Survived'] + sub5['Survived'] + sub6['Survived'])/10
sub.Survived = np.where(res > 0.5, 1, 0).astype(int)

sub.to_csv("C:/data/keggle/taita/answer/submission6_0429.csv", index = False)
sub['Survived'].mean()


