import sys
import random
import re

def main(in_file, splits):
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
    
    with open(in_file.replace(".txt", ".train.txt"), "w") as output_train:
        output_train.write("\n".join(train_list))

    with open(in_file.replace(".txt", ".dev.txt"), "w") as output_dev:
        output_dev.write("\n".join(dev_list))
        
    with open(in_file.replace(".txt", ".eval.txt"), "w") as output_test:
        output_test.write("\n".join(test_list))
    
    regexp = re.compile(r"\$.+", flags=re.MULTILINE)
    out_list_clean = [re.sub(regexp, "", example) for example in test_list]
    punctuation_pattern = re.compile('["·?¡¿!,:;]')
    

if __name__ == '__main__':
    random.seed(1234)
    splits = [0.8, 0.1, 0.1]
    assert sum(splits) == 1, "Please verify proportions"
    main(sys.argv[1], splits)