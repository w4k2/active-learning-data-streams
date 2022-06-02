import numpy as np


def q_statistic_sequence(classifier_preds, targets, unsupervised=False):
    result = []
    for i in range(1, classifier_preds.shape[0]):
        if unsupervised:
            q_stat = q_statistic_unsupervised(classifier_preds[i])
        else:
            q_stat = q_statistic(classifier_preds[i], targets)
        result.append(q_stat)
    return result


def q_statistic(classifier_preds, targets):
    num_classifiers = classifier_preds.shape[0]
    num_samples = classifier_preds.shape[1]
    assert len(targets) == num_samples

    pairwise_Q_stats = []

    for i in range(num_classifiers):
        for j in range(i+1, num_classifiers):
            first_classifier_preds = classifier_preds[i, :]
            second_classifier_preds = classifier_preds[j, :]

            targets = np.reshape(targets, newshape=first_classifier_preds.shape)
            first_correct_preds = first_classifier_preds == targets
            second_correct_preds = second_classifier_preds == targets

            both_correct = first_correct_preds == second_correct_preds
            first_correct = first_correct_preds == np.logical_not(second_correct_preds)
            second_correct = np.logical_not(first_correct_preds) == second_correct_preds
            both_incorrect = np.logical_not(first_correct_preds) == np.logical_not(second_correct_preds)

            N11 = np.sum(both_correct)
            N10 = np.sum(first_correct)
            N01 = np.sum(second_correct)
            N00 = np.sum(both_incorrect)

            assert N11 + N10 + N01 + N00 == 2 * num_samples

            Q_stat = (N11 * N00 - N01 * N10) / (N11 * N00 + N01 * N10)
            pairwise_Q_stats.append(Q_stat)

    Q_av = 2 / num_classifiers / (num_classifiers - 1) * sum(pairwise_Q_stats)

    return Q_av


def q_statistic_unsupervised(classifier_preds):
    num_classifiers = classifier_preds.shape[0]
    num_samples = classifier_preds.shape[1]

    pairwise_Q_stats = []

    for i in range(num_classifiers):
        for j in range(i+1, num_classifiers):
            first_classifier_preds = classifier_preds[i, :]
            second_classifier_preds = classifier_preds[j, :]

            agree = first_classifier_preds == second_classifier_preds
            disagree = first_classifier_preds != second_classifier_preds

            n_agree = np.sum(agree)
            n_disagree = np.sum(disagree)

            assert n_agree + n_disagree == num_samples

            Q_stat = n_agree / num_samples
            pairwise_Q_stats.append(Q_stat)

    Q_av = 2 / num_classifiers / (num_classifiers - 1) * sum(pairwise_Q_stats)

    return Q_av
