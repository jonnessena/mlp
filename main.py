import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import minmax_scale

# removing useless warning messages...
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# (portuguese) comentários em português nessa função de leitura da planilha
def read_dataset(filename, lookback_window=10, forecast_window=12, 
                 timeframe = None, interval=['11/03/1973', '15/04/2019']):
    # (portuguese) Na planilha, as datas estão no formato BR.
    dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%Y')
    serie = pd.read_excel(filename, parse_dates=[0], date_parser=dateparse)

    dataframe = pd.DataFrame(serie)
    
    # (portuguese) considerar somente valores entre as datas 11/03/1973 e 
    # 02/10/2018 porque as linhas da tabela estão completas neste intervalo.  
    date1 = pd.datetime.strptime(interval[0], '%d/%m/%Y') 
    date2 = pd.datetime.strptime(interval[1], '%d/%m/%Y')
    dataframe = dataframe.loc[(dataframe['Unnamed: 0']>date1) & 
                              (dataframe['Unnamed: 0']<date2)]
    if(timeframe is not None):
        exit() # (portuguese) adicionar código para as considerações 4 e 5 que
               # tratam de timeframe NOVEMBRO a ABRIL e MAIO a OUTUBRO
    dataframe.fillna(method='pad')  # (portuguese) preenche valores nulos por 
                                    # valores passados 
    print("Dataset: " + filename)
    print(dataframe.head())
    
    # (portuguese) converter tabela para uma matriz numpy...
    data = dataframe.values
    data = np.concatenate([data])

    data = data[:,1:] # (portuguese) remove a primeira coluna com as datas.
    print("Data length: ", data.shape)
    
    data = minmax_scale(data, feature_range=(0, 1)) # (portuguese) normalização  
                                                    # dos dados entre 0 e 1

    # (portuguese) ordenação dos valores passados da planilha
    # X = data[:,60-lookback_window:60] = versão monovariada (somente UBE)
    slices = [(10-lookback_window, 10), # ATP
             (20-lookback_window, 20), # RBG
             (30-lookback_window, 30), # BLS
             (40-lookback_window, 40), # SFB
             (50-lookback_window, 50), # FZB
             (60-lookback_window, 60)] # UBE (valores passados)
    X = np.hstack(data[:,low:high] for low, high in slices)
    y = data[:, data.shape[1]-forecast_window:]
    return X, y


filename = 'UBE_DADOS_DEFASADOS_TREINO_EXERCICIO_01-03-1973_2019.xls'

X, y = read_dataset(filename,7) # (portuguese) integrante 1: considerar 7  
                                # valores passados de cada variável (48  
                                # entradas) e 12 saídas;

train_partition = 0.75
validation_partition = 0.15  # 0.2*train
test_partition = 0.1

X_train = X[:int(X.shape[0]*(train_partition+validation_partition))]
y_train = y[:int(y.shape[0]*(train_partition+validation_partition))]

X_test = X[X_train.shape[0]:]
y_test = y[y_train.shape[0]:]

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import initializers
from keras.callbacks import EarlyStopping

# Traditional mlp implementation in Keras: [Valença, 2007] Valença, Mêuser J. S.
# 2007. Fundamentos das redes neurais: exemplos em Java. Olinda/PE: Livro 
# Rápido, 1a Edição, 382 páginas.
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
model = Sequential()
# single hidden layer with 50 neurons and tanh activation
model.add(Dense(50, input_dim=input_dim, activation='tanh', 
                kernel_initializer=initializers.glorot_normal(seed=160391)))
# output layer and sigmoid activation
model.add(Dense(output_dim, activation='sigmoid',
                kernel_initializer=initializers.glorot_normal(seed=160391)))

# RMSprop optimization: https://www.youtube.com/watch?v=5Yt-obwvMHI
optimizer = optimizers.RMSprop(lr=5e-3)

# (portuguese) métrica da ONS: avaliação do neurônio 4 até o 10
def ONS_mse(y_true, y_pred):
    return tf.losses.mean_squared_error(y_true[3:10], y_pred[3:10])

model.compile(optimizer=optimizer, loss='mse', metrics=['mape', ONS_mse])

print('Train...')
# cross-validation
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
# fast and effective training configuration 
history = model.fit(X, y, epochs=100, batch_size=1024, validation_split=0.2, 
                    callbacks=[es], verbose=0)

# plotting trainning results...
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.head())

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.plot(hist['epoch'], hist['loss'], label='Train Error')
plt.plot(hist['epoch'], hist['val_loss'], label='Validation Error')
plt.title('Training graph:')
plt.legend()
plt.show()

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Percentage Error')
plt.plot(hist['epoch'], hist['mean_absolute_percentage_error'], 
         label='Train Error')
plt.title('Training graph:')
plt.legend()
plt.show()

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('ONS Mean Squared Error')
plt.plot(hist['epoch'], hist['ONS_mse'], label='Train Error')
plt.title('Training graph:')
plt.legend()
plt.show()

# testing and plotting model forecasting performance
p = model.predict(X_test)
plt.plot(p[:,0])
plt.plot(y_test[:,0])
if len(filename)> 20:
    filename = filename[0:20]
plt.title(filename+": One day forecasting for test dataset")
plt.legend(['predicted', 'actual'])
plt.show()

# get test rmse
RMSE = tf.sqrt(tf.losses.mean_squared_error(p, y_test)) # (portuguese) Erro
                                                        # médio quadrático.
with tf.Session() as sess:
    result = sess.run(RMSE)

temp = "RMSE: %.2f"%np.mean(result)
print(temp)
