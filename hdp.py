{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 123,
   "id": "ca9b6452",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score\n",
    "import seaborn as sn\n",
    "from sklearn.metrics import mean_squared_error\n",
    "from sklearn.metrics import mean_absolute_error\n",
    "import pickle "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "id": "1a0e1c2e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>cp</th>\n",
       "      <th>trestbps</th>\n",
       "      <th>chol</th>\n",
       "      <th>fbs</th>\n",
       "      <th>restecg</th>\n",
       "      <th>thalach</th>\n",
       "      <th>exang</th>\n",
       "      <th>oldpeak</th>\n",
       "      <th>slope</th>\n",
       "      <th>ca</th>\n",
       "      <th>thal</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>63</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>145</td>\n",
       "      <td>233</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>150</td>\n",
       "      <td>0</td>\n",
       "      <td>2.3</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>37</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>130</td>\n",
       "      <td>250</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>187</td>\n",
       "      <td>0</td>\n",
       "      <td>3.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>41</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>130</td>\n",
       "      <td>204</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>172</td>\n",
       "      <td>0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>56</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>120</td>\n",
       "      <td>236</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>178</td>\n",
       "      <td>0</td>\n",
       "      <td>0.8</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>57</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>120</td>\n",
       "      <td>354</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>163</td>\n",
       "      <td>1</td>\n",
       "      <td>0.6</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>298</th>\n",
       "      <td>57</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>140</td>\n",
       "      <td>241</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>123</td>\n",
       "      <td>1</td>\n",
       "      <td>0.2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>299</th>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>110</td>\n",
       "      <td>264</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>132</td>\n",
       "      <td>0</td>\n",
       "      <td>1.2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>300</th>\n",
       "      <td>68</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>144</td>\n",
       "      <td>193</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>141</td>\n",
       "      <td>0</td>\n",
       "      <td>3.4</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>301</th>\n",
       "      <td>57</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>130</td>\n",
       "      <td>131</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>115</td>\n",
       "      <td>1</td>\n",
       "      <td>1.2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>302</th>\n",
       "      <td>57</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>130</td>\n",
       "      <td>236</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>174</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>303 rows × 14 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  \\\n",
       "0     63    1   3       145   233    1        0      150      0      2.3   \n",
       "1     37    1   2       130   250    0        1      187      0      3.5   \n",
       "2     41    0   1       130   204    0        0      172      0      1.4   \n",
       "3     56    1   1       120   236    0        1      178      0      0.8   \n",
       "4     57    0   0       120   354    0        1      163      1      0.6   \n",
       "..   ...  ...  ..       ...   ...  ...      ...      ...    ...      ...   \n",
       "298   57    0   0       140   241    0        1      123      1      0.2   \n",
       "299   45    1   3       110   264    0        1      132      0      1.2   \n",
       "300   68    1   0       144   193    1        1      141      0      3.4   \n",
       "301   57    1   0       130   131    0        1      115      1      1.2   \n",
       "302   57    0   1       130   236    0        0      174      0      0.0   \n",
       "\n",
       "     slope  ca  thal  target  \n",
       "0        0   0     1       1  \n",
       "1        0   0     2       1  \n",
       "2        2   0     2       1  \n",
       "3        2   0     2       1  \n",
       "4        2   0     2       1  \n",
       "..     ...  ..   ...     ...  \n",
       "298      1   0     3       0  \n",
       "299      1   0     3       0  \n",
       "300      1   2     3       0  \n",
       "301      1   1     3       0  \n",
       "302      1   1     2       0  \n",
       "\n",
       "[303 rows x 14 columns]"
      ]
     },
     "execution_count": 112,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "heartdata=pd.read_csv('Desktop\\project\\heart.csv')\n",
    "heartdata"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "id": "5c2368bb",
   "metadata": {},
   "outputs": [],
   "source": [
    "X=heartdata.drop(['target'],axis=1)\n",
    "y=heartdata['target']\n",
    "xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "id": "7b758e24",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DecisionTreeClassifier(random_state=0)"
      ]
     },
     "execution_count": 114,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model=DecisionTreeClassifier(random_state=0)\n",
    "model.fit(xtrain,ytrain)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "id": "1f901af8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1,\n",
       "       0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0,\n",
       "       1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1], dtype=int64)"
      ]
     },
     "execution_count": 115,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "prediction=model.predict(xtest)\n",
    "prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 116,
   "id": "8017e0fd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 88.52459016393442 %\n",
      "Precision: 82.35294117647058 %\n",
      "Recall: 96.55172413793103 %\n",
      "F1 score: 88.8888888888889 %\n"
     ]
    }
   ],
   "source": [
    "print('Accuracy:',accuracy_score(ytest,prediction)*100,'%')\n",
    "print('Precision:',precision_score(ytest,prediction)*100,'%')\n",
    "print('Recall:',recall_score(ytest,prediction)*100,'%')\n",
    "print('F1 score:',f1_score(ytest,prediction)*100,'%')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 117,
   "id": "7efd9419",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Predicted', ylabel='Actual'>"
      ]
     },
     "execution_count": 117,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "confusion_matrix=pd.crosstab(ytest,prediction,rownames=['Actual'],colnames=['Predicted'])\n",
    "sn.heatmap(confusion_matrix,annot=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 118,
   "id": "dce1a169",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MSE= 0.11475409836065574\n",
      "MAE= 0.11475409836065574\n"
     ]
    }
   ],
   "source": [
    "print('MSE=',mean_squared_error(ytest,prediction))\n",
    "print('MAE=',mean_absolute_error(ytest,prediction))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 130,
   "id": "d2c6283c",
   "metadata": {},
   "outputs": [],
   "source": [
    "number_input=[]\n",
    "def predicting(number_input):\n",
    "    target=(model.predict([number_input]))\n",
    "    if target==1:\n",
    "        print(\"You may have heart disease.\")\n",
    "    elif target==0:\n",
    "        print(\"You may not have heart disease.\")\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "id": "2deec4a8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "You may have heart disease.\n"
     ]
    }
   ],
   "source": [
    "predicting([43,1,2,145,233,1,0,150,0,2.3,2,0,1])"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
