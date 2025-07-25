import time
import joblib
import seaborn as sns
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression

matplotlib.use("TkAgg")


class main_model:

    def Dataset_Model(self):
        try:
            stop_words = set(stopwords.words("english"))

            data = pd.read_csv("data/SMSSpamCollection", sep="\t", header=None, names=["label", "text"],
                               encoding="latin-1")
            print("Succesfully Read the Dataset .")
            data.columns = ["label", "text"]

            data["label"] = data["label"].map({"ham": 0, "spam": 1})

            x_train_text, x_test_text, y_train, y_test = train_test_split(
                data["text"], data["label"], test_size=0.3, random_state=42, stratify=data["label"]
            )

            vectorizer = TfidfVectorizer(stop_words="english")
            x_train_vec = vectorizer.fit_transform(x_train_text)
            x_test_vec = vectorizer.transform(x_test_text)

            sm = SMOTE(random_state=42)
            x_train_balanced, y_train_balanced = sm.fit_resample(x_train_vec, y_train)

            model = LogisticRegression()
            model.fit(x_train_balanced, y_train_balanced)

            y_pred = model.predict(x_test_vec)
            print("Accuracy:", accuracy_score(y_test, y_pred))
            print(classification_report(y_test, y_pred))

            confusion = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(confusion, annot=True, fmt='d', cmap="Blues", xticklabels=["ham", "spam"],
                        yticklabels=["ham", "spam"])

            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.grid(True)
            plt.savefig("Confusion_Matrix.png")
            plt.show()
            plt.close()

            time.sleep(random.randint(1,2))

            try:
                joblib.dump(model, "Model/Spam_Safe.pkl")
                joblib.dump(vectorizer, "Model/vectorizer.pkl")
                print("Success Model Saved Succesfully .")
            except Exception as e:
                print(f"Failed to Save Models {e}")

        except Exception as e:
            print(f"Failed to Train Model due to :{e}")


C = main_model()
C.Dataset_Model()
