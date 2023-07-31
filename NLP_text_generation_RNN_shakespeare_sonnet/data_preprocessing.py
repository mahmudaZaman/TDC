from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import string

def text_cleaner(text):
    text = "".join(car for car in text if car not in string.punctuation).lower()
    text = text.encode("utf8").decode("ascii",'ignore')
    return text

def get_lines(file_name):
  get_file = open(file_name, "rb")
  lines = get_file.read().decode("utf-8")
  return lines


def calculate_results(y_true, y_pred):
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results