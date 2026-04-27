import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

DATASET_PATH = "dataset/"
BATCH_SIZE = 32
IMG_HEIGHT = 120
IMG_WIDTH = 120


def load_and_split_data():

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    val_ds_initial = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    # Podział 20% walidacji na 10% walidacji i 10% testów
    val_batches = tf.data.experimental.cardinality(val_ds_initial)
    test_ds = val_ds_initial.take(val_batches // 2)
    val_ds = val_ds_initial.skip(val_batches // 2)

    return train_ds, val_ds, test_ds


def build_model():
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Trening')
    plt.plot(epochs_range, val_acc, label='Walidacja')
    plt.legend(loc='lower right')
    plt.title('Dokładność')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Trening')
    plt.plot(epochs_range, val_loss, label='Walidacja')
    plt.legend(loc='upper right')
    plt.title('Strata')

    plt.savefig('wykres_uczenia.png')
    plt.show()

def evaluate_model(model, test_ds):
    print("\nOcena na zbiorze testowym: ")
    loss, accuracy = model.evaluate(test_ds)
    print(f"Skuteczność na danych testowych: {accuracy * 100:.2f}%")

    # Zbieranie etykiet (prawdziwych i przewidzianych przez model)
    y_true =[]
    y_pred_probs =[]

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred_probs.extend(preds)

    y_true = np.array(y_true)
    y_pred = (np.array(y_pred_probs) > 0.5).astype(int) # Próg 0.5 dla klasyfikacji

    # Raport klasyfikacji
    print("\nRaport Klasyfikacji:")
    print(classification_report(y_true, y_pred, target_names=['Brak pęknięcia (Negative)', 'Pęknięcie (Positive)']))

    # Rysowanie Macierzy Pomyłek (Confusion Matrix)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Brak pęknięcia', 'Pęknięcie'],
                yticklabels=['Brak pęknięcia', 'Pęknięcie'])
    plt.title('Macierz Pomyłek (Confusion Matrix)')
    plt.ylabel('Prawdziwa etykieta')
    plt.xlabel('Przewidziana etykieta')
    plt.savefig('macierz_pomylek.png')
    plt.show()

def main():
    train_ds, val_ds, test_ds = load_and_split_data()


    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model = build_model()

    print("Starting training")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10
    )

    model.save('crack_detection.h5')
    plot_history(history)
    evaluate_model(model, test_ds)

if __name__ == "__main__":
    main()