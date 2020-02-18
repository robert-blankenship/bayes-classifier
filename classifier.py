from typing import List, Set
import numpy as np


class Sample:
    def __init__(self, raw_sample: str):
        chunks = raw_sample.split('","')
        self.rating = int(chunks[0].replace('"', ""))
        assert(self.rating >= 0)
        assert(self.rating <= 4)

        self.is_negative = self.rating == 0
        self.is_positive = not self.is_negative

        self.review_text = chunks[-1].replace("\n", '')
        self.review_tokens = get_tokens(self.review_text)

    def to_string(self):
        # return "text={},tokens={},rating={}".format(self.review_text, self.review_tokens, self.rating)
        return str(self.review_tokens)

    # TODO: Performance
    def word_vec(self, vocab_list: List[str]) -> List[int]:
        vec = [0 for _ in vocab_list]
        tokens = set(self.review_tokens)
        for i, vocab in enumerate(vocab_list):
            if vocab in tokens:
                vec[i] = 1
        return vec

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()


def get_tokens(input_text: str) -> list:
    cleaned_text = input_text
    cleaned_text = cleaned_text.replace(".", "")
    cleaned_text = cleaned_text.replace("!", "")
    cleaned_text = cleaned_text.replace("'", "")
    cleaned_text = cleaned_text.replace('"', "")
    cleaned_text = cleaned_text.replace('\n', "")
    cleaned_text = cleaned_text.replace('  ', "")
    tokens = cleaned_text.split(" ")
    return [token for token in tokens if not (token.startswith("@") or token.startswith("http") or token == "")]


def load_samples(filename: str):
    samples = []
    with open(filename, "r") as training:
        for line in training:
            samples.append(Sample(line))
    return samples


def create_vocab_list(samples: list):
    vocab_set = set()
    for sample in samples:
        for token in sample.review_tokens:
            vocab_set.add(token)
    return list(vocab_set)


def main():
    training_samples = load_samples("training.csv")
    vocab_list = create_vocab_list(training_samples)
    model = train(training_samples, vocab_list)

    false_positive = 0
    false_negative = 0
    true_positive = 0
    true_negative = 0

    results = []

    for sample in training_samples:
        results.append((model.classify(sample.word_vec(vocab_list)), sample.is_positive))

    for actual, expected in results:
        if actual == expected:
            if expected is True:
                true_positive += 1
            else:
                true_negative += 1
        else:
            if expected is True:
                false_negative += 1
            else:
                false_positive += 1

    print("false_positive={}".format(false_positive))
    print("false_negative={}".format(false_negative))
    print("true_positive={}".format(true_positive))
    print("true_negative={}".format(true_negative))

    print("Accuracy: {}".format((true_negative + true_positive) / (1.0 * len(training_samples))))




class Model:
    def __init__(self, positive_probabilities, negative_probabilities, percent_negative, percent_positive):
        self.positive_probabilities = positive_probabilities
        self.negative_probabilities = negative_probabilities
        self.percent_negative = percent_negative
        self.percent_positive = percent_positive

    # Return "True" for positive sentiment, "False" for negative sentiment.
    def classify(self, word_vector):
        prob_positive = sum(word_vector * self.positive_probabilities)
        prob_negative = sum(word_vector * self.negative_probabilities)

        prob_positive += np.log(self.percent_positive)
        prob_negative += np.log(self.percent_negative)

        if prob_positive > prob_negative:
            return True
        else:
            return False


def train(samples, vocab_list):
    num_negative = len([sample for sample in samples if sample.rating == 0])
    num_positive = len(samples) - num_negative

    percent_negative = num_negative / float(len(samples))
    percent_positive = num_positive / float(len(samples))

    # positive_word_counts = np.zeros(len(vocab_list))
    positive_word_counts = np.ones(len(vocab_list))
    positive_probabilities_denom = 2.

    # negative_word_counts = np.zeros(len(vocab_list))
    negative_word_counts = np.ones(len(vocab_list))
    negative_probabilities_denom = 2.

    for sample in samples:
        if sample.is_negative:
            negative_word_counts += sample.word_vec(vocab_list)
            negative_probabilities_denom += len(sample.review_tokens)
        else:
            positive_word_counts += sample.word_vec(vocab_list)
            positive_probabilities_denom += len(sample.review_tokens)

    # TODO: Why the log?
    positive_probabilities = np.log(positive_word_counts / positive_probabilities_denom)
    negative_probabilities = np.log(negative_word_counts / negative_probabilities_denom)

    return Model(positive_probabilities=positive_probabilities,
                 negative_probabilities=negative_probabilities,
                 percent_positive=percent_positive,
                 percent_negative=percent_negative)


if __name__ == "__main__":
    main()
