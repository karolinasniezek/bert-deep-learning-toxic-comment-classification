import sys
import os

print(sys.executable)
print("TF_USE_LEGACY_KERAS =", os.environ.get("TF_USE_LEGACY_KERAS"))
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix


# 1. Stała wartość seed
seed = 42

# 2. Ustawienie seedów dla reprodukowalności
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.keras.utils.set_random_seed(seed)

# 3. Wymuszenie deterministycznych operacji w TensorFlow
os.environ["TF_DETERMINISTIC_OPS"] = "1"
tf.config.experimental.enable_op_determinism()

bert_mini_path = "./bert-tiny"

df = pd.read_csv("../data/toxic_subset.csv")


texts = df["comment_text"].tolist()
labels = df[["toxic", "severe_toxic", "obscene", 'threat', "insult", "identity_hate"]].values

# tokenizer = BertTokenizer.from_pretrained("prajjwall/bert-tiny")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

texts_train_val , texts_test, y_train_val, y_test = train_test_split(texts, labels, test_size=0.1, random_state=seed)
texts_train, texts_val, y_train, y_val = train_test_split(texts_train_val, y_train_val, test_size=0.2, random_state=seed)

max_len = 128

X_train = tokenizer(
    texts_train,
    max_length=max_len,
    truncation=True,
    padding="max_length",
    return_tensors="tf"
)

X_val = tokenizer(
    texts_val,
    max_length=max_len,
    truncation=True,
    padding="max_length",
    return_tensors="tf"
)

X_test = tokenizer(
    texts_test,
    max_length=max_len,
    truncation=True,
    padding="max_length",
    return_tensors="tf"
)

def build_model():
    bert = TFBertModel.from_pretrained("bert-base-uncased", from_pt=True)
    input_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

    bert_output = bert(input_ids, attention_mask=attention_mask)[1]

    x = tf.keras.layers.Dropout(0.3, seed=seed)(bert_output)
    x = tf.keras.layers.Dense(
        64,
        activation="leaky_relu",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
    )(x)
    x = tf.keras.layers.Dense(
        6,
        activation="sigmoid",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
    )(x)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=x)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# model = build_model()

# history = model.fit(
#     {
#         "input_ids": X_train["input_ids"],
#         "attention_mask": X_train["attention_mask"]
#     },
#     y_train,
#     validation_data=(
#         {
#             "input_ids": X_val["input_ids"],
#             "attention_mask": X_val["attention_mask"]
#         },
#         y_val
#     ),
#     epochs=3,
#     batch_size=32
# )

model = build_model()
model.load_weights("toxic_model.weights.h5")

model.save("toxic_model.h5", include_optimizer=False)



loss, acc = model.evaluate(
    {"input_ids": X_test["input_ids"], "attention_mask": X_test["attention_mask"]},
    y_test
)
print(f"\nTest accuracy: {acc: .4f}")

# plt.figure(figsize=(8, 5))
#
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
#
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Training vs Validation Accuracy')
# plt.legend()
#
# plt.savefig("accuracy_plot.png")
#
# plt.show()

model.save_weights("toxic_model.weights.h5")

y_pred_probs = model.predict(
    {
        "input_ids": X_test["input_ids"],
        "attention_mask": X_test["attention_mask"]
    }
)

y_pred = (y_pred_probs > 0.5).astype(int)

cm = multilabel_confusion_matrix(y_test, y_pred)

with open("macierz_pomylek.txt", "w") as f:
    for numer_klasy, macierz in enumerate(cm):
        tekst = f"Klasa {numer_klasy}:\n{macierz}\n\n"
        print(tekst)
        f.write(tekst)


