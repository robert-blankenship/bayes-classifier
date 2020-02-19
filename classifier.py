#!/usr/bin/env python3
import heapq
import numpy as np
import sys
from typing import List, Set


class Vocab:
    def __init__(self, vocab_list: List[str]):
        self.list = vocab_list
        self.length = len(vocab_list)
        self.indices = {}
        for i, el in enumerate(vocab_list):
            self.indices[el] = i


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
    def word_vec(self, vocab: Vocab) -> np.array:
        vec = np.zeros(len(vocab.list))

        for token in self.review_tokens:
            if token in vocab.indices:
                vec[vocab.indices[token]] = 1

        return vec

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()


def get_tokens(input_text: str) -> List[str]:
    cleaned_text = input_text
    cleaned_text = cleaned_text.replace(".", "")
    cleaned_text = cleaned_text.replace("!", "")
    cleaned_text = cleaned_text.replace("'", "")
    cleaned_text = cleaned_text.replace('"', "")
    cleaned_text = cleaned_text.replace('\n', "")
    cleaned_text = cleaned_text.replace('  ', "")
    tokens = cleaned_text.split(" ")
    return [token for token in tokens if not (token.startswith("@") or token.startswith("http") or token == "")]


def load_samples(filename: str, encoding='ISO-8859-1', max_size=sys.maxsize,
                 test_percentage: float = .1) -> (List[Sample], List[Sample]):
    samples = []
    with open(filename, "r", encoding=encoding) as training:
        for line in training:
            samples.append(Sample(line))

    # Shuffle the samples.
    np.random.seed(42)
    np.random.shuffle(samples)

    # Use only a maximum number of samples.
    samples = samples[0:int(max_size)]

    training_samples = samples[0:int(len(samples) * (1 - test_percentage))]
    testing_samples = samples[-int(len(samples) * test_percentage):]

    return training_samples, testing_samples


def create_vocab_list(samples: list):
    vocab_set = set()
    for sample in samples:
        for token in sample.review_tokens:
            vocab_set.add(token)
    return list(vocab_set)


def create_vocab_max_size(samples: list, max_vocab_size=float('inf')):
    vocab_heap = []

    vocab_counts = {}
    for sample in samples:
        for token in sample.review_tokens:
            if token not in vocab_counts:
                vocab_counts[token] = 1
            else:
                vocab_counts[token] += 1

    words_removed = 0

    for item, count in vocab_counts.items():
        heapq.heappush(vocab_heap, (count, item))

        while len(vocab_heap) > max_vocab_size:
            heapq.heappop(vocab_heap)
            words_removed += 1

    print("Removed {} word(s) from the vocabulary".format(words_removed))

    return Vocab([word for count, word in vocab_heap])


def main():
    print("Loading samples")
    training_samples, testing_samples = load_samples("data/training.1600000.processed.noemoticon.csv", max_size=sys.maxsize)

    print("Creating vocab")
    vocab = create_vocab_max_size(training_samples, max_vocab_size=5000)
    print("Created vocab: length={}".format(vocab.length))

    print("Training model")
    model = train(training_samples, vocab)

    print("Testing accuracy")
    test(test_samples=testing_samples, model=model, vocab=vocab)


def test(test_samples, model, vocab: Vocab, reporting_periods=10):
    reporting_interval = len(test_samples) / reporting_periods

    false_positive = 0
    false_negative = 0
    true_positive = 0
    true_negative = 0

    results = []

    for i, sample in enumerate(test_samples):
        results.append((model.classify(sample.word_vec(vocab)), sample.is_positive))
        if i % reporting_interval == 0:
            print("Tested {}/{} samples".format(i, len(test_samples)))

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

    print("Accuracy: {}".format((true_negative + true_positive) / (1.0 * len(test_samples))))


class Model:
    def __init__(self, positive_probabilities: np.array, negative_probabilities: np.array,
                 percent_negative: float, percent_positive: float):
        self.positive_probabilities: np.array = positive_probabilities
        self.negative_probabilities: np.array = negative_probabilities
        self.percent_negative: float = percent_negative
        self.percent_positive: float = percent_positive

    # Return "True" for positive sentiment, "False" for negative sentiment.
    def classify(self, word_vector: np.array):
        prob_positive = np.sum(word_vector * self.positive_probabilities)
        prob_negative = np.sum(word_vector * self.negative_probabilities)

        prob_positive += np.log(self.percent_positive)
        prob_negative += np.log(self.percent_negative)

        if prob_positive > prob_negative:
            return True
        else:
            return False


def train(samples, vocab: Vocab, reporting_periods=10):
    # Log each time this number of samples have been trained.
    training_reporting_period = len(samples) / reporting_periods

    num_negative = len([sample for sample in samples if sample.rating == 0])
    num_positive = len(samples) - num_negative

    percent_negative = num_negative / float(len(samples))
    percent_positive = num_positive / float(len(samples))

    # positive_word_counts = np.zeros(len(vocab_list))
    positive_word_counts = np.ones(vocab.length)
    positive_probabilities_denom = 2.

    # negative_word_counts = np.zeros(len(vocab_list))
    negative_word_counts = np.ones(vocab.length)
    negative_probabilities_denom = 2.

    for i, sample in enumerate(samples):
        if sample.is_negative:
            negative_word_counts += sample.word_vec(vocab)
            negative_probabilities_denom += len(sample.review_tokens)
        else:
            positive_word_counts += sample.word_vec(vocab)
            positive_probabilities_denom += len(sample.review_tokens)

        if i % training_reporting_period == 0:
            print("Trained {}/{} samples".format(i, len(samples)))

    # TODO: Why the log?
    positive_probabilities = np.log(positive_word_counts / positive_probabilities_denom)
    negative_probabilities = np.log(negative_word_counts / negative_probabilities_denom)

    return Model(positive_probabilities=positive_probabilities,
                 negative_probabilities=negative_probabilities,
                 percent_positive=percent_positive,
                 percent_negative=percent_negative)


if __name__ == "__main__":
    main()
