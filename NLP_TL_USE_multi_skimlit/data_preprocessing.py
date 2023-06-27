from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def get_lines(file_name):
  get_file = open(file_name, 'r')
  lines = get_file.readlines()
  return lines

def text_data_preprocess(file_name):
  abstruct_sentences = get_lines(file_name)
  abstruct_data = []
  abstruct_lines = []

  for sentence in abstruct_sentences:
    if sentence.startswith("###"):
      abstract_id = sentence
      abstruct_lines = []
    elif sentence.isspace():
      for abstruct_line_number, abstruct_line in enumerate(abstruct_lines):
        line_data = {}
        abstruct_line_split = abstruct_line.split("\t")
        line_data["target"] = abstruct_line_split[0]
        line_data["text"] = abstruct_line_split[1].lower().strip()
        line_data["line_number"] = abstruct_line_number
        line_data["total_lines"] =len(abstruct_lines)-1
        abstruct_data.append(line_data)
    else:
        abstruct_lines.append(sentence)
  return abstruct_data


def calculate_results(y_true, y_pred):
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results