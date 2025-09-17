def label_to_int(string_label):
    if string_label == 'gotasH2O': return 1
    if string_label == 'lunas': return 2
    if string_label == 'rayos':
        return 3

    else:
        raise Exception('unkown class_label')


def int_to_label(string_label):
    if string_label == 1: return 'gotasH2O'
    if string_label == 2: return 'lunas'
    if string_label == 3:
        return 'rayos'
    else:
        raise Exception('unkown class_label')