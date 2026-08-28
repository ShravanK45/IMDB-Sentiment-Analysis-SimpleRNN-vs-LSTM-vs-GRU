
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Dense,SimpleRNN,LSTM,GRU

#Load Dataset
vocab_size=10000
max_len=200
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=vocab_size)

x_train=pad_sequences(x_train,maxlen=max_len,padding='post')
x_test=pad_sequences(x_test,maxlen=max_len,padding='post')

print(f"Training Shape:{x_train.shape}")
print(f"Test Data Shape:{x_test.shape}")

#Define and Train SimpleRNN Model
rnn_model=Sequential([
    Embedding(input_dim=vocab_size,output_dim=128),
    SimpleRNN(128,activation="tanh"),
    Dense(1,activation='sigmoid')
])

rnn_model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
rnn_model.summary()

rnn_history=rnn_model.fit(x_train,y_train,validation_split=0.2,batch_size=64,verbose=1,epochs=3)
rnn_loss,rnn_accuracy=rnn_model.evaluate(x_test,y_test)
print(f"RNN Model Loss:{rnn_loss}, RNN Model Accuracy:{rnn_accuracy}")

#Define adn Train LSTM Model
lstm_model=Sequential([
    Embedding(input_dim=vocab_size,output_dim=128),
    LSTM(128,activation="tanh"),
    Dense(1,activation='sigmoid')
])

lstm_model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
lstm_model.summary()

lstm_history=lstm_model.fit(x_train,y_train,validation_split=0.2,batch_size=64,verbose=1,epochs=3)
lstm_loss,lstm_accuracy=lstm_model.evaluate(x_test,y_test)
print(f"LSTM Model Loss:{lstm_loss}, LSTM Model Accuracy:{lstm_accuracy}")

#Define adn Train GRU Model
gru_model=Sequential([
    Embedding(input_dim=vocab_size,output_dim=128),
    GRU(128,activation="tanh"),
    Dense(1,activation='sigmoid')
])

gru_model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
gru_model.summary()

gru_history=gru_model.fit(x_train,y_train,validation_split=0.2,batch_size=64,verbose=1,epochs=3)
gru_loss,gru_accuracy=gru_model.evaluate(x_test,y_test)
print(f"GRU Model Loss:{gru_loss}, GRU Model Accuracy:{gru_accuracy}")

#Comparing Models
results = {
    "SimpleRNN": {
        "loss": rnn_loss,
        "accuracy": rnn_accuracy
    },
    "LSTM": {
        "loss": lstm_loss,
        "accuracy": lstm_accuracy
    },
    "GRU": {
        "loss": gru_loss,
        "accuracy": gru_accuracy
    }
}

for model_name, metrics in results.items():
    print(
        f"{model_name} | "
        f"Loss: {metrics['loss']:.4f} | "
        f"Accuracy: {metrics['accuracy']:.4f}"
    )

#PLOT Training Accuracy
import matplotlib.pyplot as plt

plt.plot(rnn_history.history['accuracy'],label='RNN Training Accuracy')
plt.plot(lstm_history.history['accuracy'],label='LSTM Training Accuracy')
plt.plot(gru_history.history['accuracy'],label='GRU Training Accuracy')
plt.legend()
plt.title("Training Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

#Plot Validation Accuracy
plt.figure(figsize=(10, 5))
plt.plot(rnn_history.history['val_accuracy'],label='RNN Validation Accuracy')
plt.plot(lstm_history.history['val_accuracy'],label='LSTM Validation Accuracy')
plt.plot(gru_history.history['val_accuracy'],label='GRU Validation Accuracy')
plt.title("Validation Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

#Accuracy Bar Plot
model_names = ["SimpleRNN", "LSTM", "GRU"]
accuracies = [rnn_accuracy,lstm_accuracy,gru_accuracy]
plt.figure(figsize=(8, 5))
plt.bar(model_names, accuracies)
plt.title("Test Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.show()