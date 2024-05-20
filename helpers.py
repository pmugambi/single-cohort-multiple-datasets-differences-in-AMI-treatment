import math


def create_combined_drugs_string(drugs):
    """

    :return:
    """
    string = ""
    for x in sorted(drugs):
        string += x + "|"
    return string[0:len(string)-1]


def create_batches(max_length, batch_size=5000):
    """

    :param max_length:
    :param batch_size:
    :return:
    """
    batch_number = math.ceil(max_length / batch_size)
    batches = []
    for i in range(1, batch_number + 1):
        if i * batch_size > max_length:
            batch_i = [(i - 1) * batch_size, max_length]
        else:
            batch_i = [(i - 1) * batch_size, i * batch_size]
        batches.append(batch_i)
    return batches


if __name__ == '__main__':
    create_combined_drugs_string(["a"])
