import sys
import random
import re
import random

def dice_roll():
    if random.random() < .5:
        result = False
    else:
        result = True
    return result


def add_noise(corpus:list):
    """
    This function randomly adds noise to a corpus. It removes the punctuation for half of it
    """
    punctuation_regexp = re.compile(r'[\.,:;?\(\)!]')
    modified_examples = []
    for example in corpus:
        example_lenght = len(example.split())
        if example_lenght > 200:
            print(example_lenght)
            print(example)
        if dice_roll():
            example = re.sub(punctuation_regexp, "", example)
            modified_examples.append(example) 
    return modified_examples
        

def main(in_file, splits, extension):
    random.seed(1234)
    with open(in_file, "r") as input_file:
        as_list = [line.replace("\n", "") for line in input_file.readlines()]
    train, dev, test = splits
    random.shuffle(as_list)
    train_list, dev_list, test_list = [], [], []
    for example in as_list:
        rand_integer = random.random()
        if rand_integer < train:
            train_list.append(example)
        elif train < rand_integer < train + test:
            dev_list.append(example)
        else:
            test_list.append(example)

    #train_list.extend(add_noise(train_list))
    #dev_list.extend(add_noise(dev_list))
    #test_list.extend(add_noise(test_list))
    #[random.shuffle(the_list) for the_list in [train_list, dev_list, test_list]]
    
    with open(in_file.replace(f".{extension}", f".train.{extension}"), "w") as output_train:
        output_train.write("\n".join(train_list))

    with open(in_file.replace(f".{extension}", f".dev.{extension}"), "w") as output_dev:
        output_dev.write("\n".join(dev_list))
        
    with open(in_file.replace(f".{extension}", f".eval.{extension}"), "w") as output_test:
        output_test.write("\n".join(test_list))
    
    regexp = re.compile(r"\$.+", flags=re.MULTILINE)
    out_list_clean = [re.sub(regexp, "", example) for example in test_list]
    punctuation_pattern = re.compile('["·?¡¿!,:;]')
    

if __name__ == '__main__':
    random.seed(1234)
    splits = [0.8, 0.1, 0.1]
    assert sum(splits) == 1, "Please verify proportions"
    extension = sys.argv[2]
    main(sys.argv[1], splits, extension)