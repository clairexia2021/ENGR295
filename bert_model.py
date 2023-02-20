import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from transformers import TFBertForSequenceClassification, BertTokenizer, InputExample, InputFeatures
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import seaborn as sns


def bert_model():
    reviews = pd.read_csv('reviews_filtered.csv')
    reviews['sentiment'] = np.where(reviews['star'] >= 6, 1, 0)

    train, validation, test = np.split(reviews.sample(frac=1), [int(.6 * len(reviews)), int(.8*len(reviews))])

   # train = reviews[:150]
   #  validation = reviews[150:200]
   #  test = reviews[200:]

    DATA_COLUMN = 'review'
    LABEL_COLUMN = 'sentiment'

    model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # example = 'In this Kaggle notebook, I will do sentiment analysis using BERT with Huggingface'
    # tokens = tokenizer.tokenize(example)
    # token_ids = tokenizer.convert_tokens_to_ids(tokens)
    # print(tokens)
    # print(token_ids)

    train_InputExamples, validation_InputExamples = convert_data_to_examples(train, validation, DATA_COLUMN, LABEL_COLUMN)

    train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
    train_data = train_data.shuffle(100).batch(32).repeat(2)

    validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
    validation_data = validation_data.batch(32)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

    model.fit(train_data, epochs=2, validation_data=validation_data)

    pred_sentences = list(test.review)
    # pred_sentences = ['worst movie of my life, will never watch movies from this series',
    #                   'Wow, blew my mind, what a movie by Marvel, animation and story is amazing']
    tf_batch = tokenizer(pred_sentences, max_length=128, padding=True, truncation=True,
                         return_tensors='tf')  # we are tokenizing before sending into our trained model
    tf_outputs = model(tf_batch)
    tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
    # axis=-1, this means that the index that will be returned by argmax
    # will be taken from the *last* axis.

    labels = ['Negative', 'Positive']
    label = tf.argmax(tf_predictions, axis=1)
    label = label.numpy()
    for i in range(len(pred_sentences)):
        print(labels[label[i]])
    test['pred'] = label.tolist()
    test.to_csv("test.csv", sep=',')
    accuray_analysis(test.sentiment, test.pred)


def accuray_analysis(actual, pred):
    sns.set(rc={'figure.figsize': (11, 5)}, font_scale=1.5, style='whitegrid')
    ax = sns.heatmap(confusion_matrix(actual, pred), annot=True, fmt="d", cmap="YlGnBu")
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predict')
    ax.set_ylabel('Condition')
    fig = ax.get_figure()
    fig.savefig("confusion_matrix.png")

    accuracy = accuracy_score(actual, pred)
    print("Test Accuracy is: %0.3f" % accuracy)
    print(classification_report(actual, pred))


def convert_data_to_examples(train, test, review, sentiment):
    train_InputExamples = train.apply(
        lambda x: InputExample(guid=None,  # Globally unique ID for bookkeeping, unused in this case
                               text_a=x[review],
                               label=x[sentiment]), axis=1)

    validation_InputExamples = test.apply(
        lambda x: InputExample(guid=None,  # Globally unique ID for bookkeeping, unused in this case
                               text_a=x[review],
                               label=x[sentiment]), axis=1, )

    return train_InputExamples, validation_InputExamples


def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = []  # -> will hold InputFeatures to be converted later

    for e in tqdm(examples):
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,    # Add 'CLS' and 'SEP'
            max_length=max_length,    # truncates if len(s) > max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True, # pads to the right by default # CHECK THIS for pad_to_max_length
            truncation=True
        )

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"], input_dict["token_type_ids"],
                                                     input_dict['attention_mask'])
        features.append(InputFeatures(input_ids=input_ids, attention_mask=attention_mask,
                                      token_type_ids=token_type_ids, label=e.label))

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )


if __name__ == '__main__':
    bert_model()
    # test = pd.read_csv('test.csv')
    # accuray_analysis(test.sentiment, test.pred)
