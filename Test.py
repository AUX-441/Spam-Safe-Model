import re
from nltk.tokenize import word_tokenize
import time
from nltk.corpus import stopwords
import joblib


class Read_And_Predict:

    def Main(self):
        try:
            model = joblib.load("Model/Spam_Safe.pkl")
            tokenizer = joblib.load("Model/vectorizer.pkl")


            print("Type exit to Quit .")
            while True:
                user_input = input("Enter Text To check :")
                if user_input.lower() == "exit":
                    print("Existing..")
                    time.sleep(1)
                    break
                stop_words = set(stopwords.words("english"))
                res = re.sub(r"[^\w\s]","",user_input)
                word = word_tokenize(res)
                filtered = [w for w in word if not w in stop_words]
                final = " ".join(filtered)

                transform = tokenizer.transform([final])
                Predicted = model.predict(transform)[0]

                label_map = f"Predicted Model for : (({final})) is = Spam" if Predicted == 1 else f"Predicted Model for (({final})) is = Safe"
                print("Predicted Model :",label_map)
        except Exception as e:
            print(f"Failed to Load Models :{e}")

C = Read_And_Predict()
C.Main()