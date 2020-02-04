{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "mat = np.random.random((100,10))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(100, 10)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mat.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "# for i in mat:\n",
    "#     print(i.dot(mat[0]) / (calcNorm(i) * calcNorm(mat[0])))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "b = np.tile(mat[0],(mat.shape[0],1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(100, 10)"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "b.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "c = (mat.dot(mat[0].T))\n",
    "c = c / (np.linalg.norm(mat, axis = 1)*calcNorm(mat[0]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1.        , 0.86022667, 0.78773655, 0.78585777, 0.77058112,\n",
       "       0.89710828, 0.84503133, 0.83058236, 0.80899688, 0.83339263,\n",
       "       0.94504951, 0.89709597, 0.90142233, 0.69078731, 0.90153473,\n",
       "       0.81754824, 0.79318203, 0.80448521, 0.89864509, 0.88870083,\n",
       "       0.91223647, 0.65893724, 0.88002879, 0.88967078, 0.76985578,\n",
       "       0.66838589, 0.80183106, 0.7356623 , 0.75689287, 0.75298618,\n",
       "       0.90858728, 0.83318243, 0.89305975, 0.77615058, 0.84299725,\n",
       "       0.85607484, 0.92042631, 0.76936163, 0.7940092 , 0.7894442 ,\n",
       "       0.64354201, 0.70692797, 0.73128558, 0.88405424, 0.6616269 ,\n",
       "       0.77757305, 0.73476856, 0.81240343, 0.87070167, 0.73842481,\n",
       "       0.59321318, 0.8396171 , 0.46243134, 0.71275426, 0.75156511,\n",
       "       0.89533578, 0.72982509, 0.62108966, 0.7464345 , 0.74964697,\n",
       "       0.83822494, 0.86559543, 0.85843107, 0.73052742, 0.84345617,\n",
       "       0.84029822, 0.81712165, 0.87888347, 0.75613781, 0.85061631,\n",
       "       0.74611047, 0.74901943, 0.871594  , 0.82940164, 0.80687383,\n",
       "       0.91958527, 0.71173988, 0.76492648, 0.64004151, 0.77121236,\n",
       "       0.7677615 , 0.81643735, 0.79633602, 0.6891363 , 0.7683798 ,\n",
       "       0.89118765, 0.86813236, 0.75351901, 0.69601148, 0.79708337,\n",
       "       0.75855346, 0.756391  , 0.74121316, 0.5974685 , 0.8003622 ,\n",
       "       0.70348188, 0.66927433, 0.85098844, 0.79162143, 0.82192301])"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1. 1. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 0. 1. 1. 1. 1. 0. 1. 1.\n",
      " 0. 0. 1. 0. 0. 0. 1. 1. 1. 0. 1. 1. 1. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 1.\n",
      " 1. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0. 0. 1. 1. 1. 0. 1. 1. 1. 1. 0. 1. 0. 0.\n",
      " 1. 1. 1. 1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0.\n",
      " 0. 1. 0. 1.]\n"
     ]
    }
   ],
   "source": [
    "# y = 1 , 0\n",
    "thresh = 0.8\n",
    "y = np.zeros((mat.shape[0]))\n",
    "y[c >= thresh] = 1\n",
    "print(y)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {},
   "outputs": [],
   "source": [
    "# help(LogisticRegression)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [],
   "source": [
    "clf = LogisticRegression(random_state=0, solver='lbfgs', C= 1000).fit(mat, y)\n",
    "y_pred = clf.predict(mat)\n",
    "# clf.predict_proba(X[:2, :]) # doctest: +ELLIPSIS\n",
    "# clf.score(y_pred, y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import confusion_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.96\n"
     ]
    }
   ],
   "source": [
    "# confusion_matrix(y_pred, y)\n",
    "\n",
    "print((y_pred == y).sum()/ len(y))\n",
    "\n",
    "# model.fit()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.fit()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "def calcNorm(x):\n",
    "    return np.sqrt((x*x).sum())\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1.4142135623730951"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a = np.array([1,1])\n",
    "calcNorm(a)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
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
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
