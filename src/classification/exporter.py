import m2cgen as m2c

from src.archived.classifier import CascadedClassifier
from src.classification.classify_entry import get_training_data


def export_model(model, lan: str='python'):
    # convert model to pure python code
    if lan=='python':
        model_to_python = m2c.export_to_python(model)
        with open("model.py", "w") as text_file:
            text_file.write(model_to_python)
    elif lan=='js':
        model_to_python = m2c.export_to_javascript(model)
        with open("model.js", "w") as text_file:
            text_file.write(model_to_python)

def get_classifier():
    training_data = get_training_data()
    # classifier = EnsembleClassifier(training_data, None)
    classifier = CascadedClassifier(training_data)
    return classifier


if __name__ == '__main__':
    clf = get_classifier().lr
    native_code = export_model(clf, 'js')
