import os
import random
import argparse


def create_sub_lists(list_path, split, seed):
    with open(list_path, "r") as file:
        paths = file.readlines()
        print(f"Read {len(paths)} lines from {list_path}")
        if seed is None:
            random.shuffle(paths)
        else:
            assert 0.0 <= float(seed) < 1.0, f"'Seed' has to be a float in [0,1)"
            random.shuffle(paths, random=lambda: float(seed))

        num_val_test = int(len(paths) * float(split)) if float(split) < 1 else int(split)
        assert len(paths) > 2 * num_val_test, f"Not enough list elements for the desired split!"
        print(f"Split value of {split} generates the following split: "
              f"{len(paths) - 2*num_val_test}:{num_val_test}:{num_val_test}")

        list_val = paths[:num_val_test]
        list_test = paths[num_val_test:2 * num_val_test]
        list_train = paths[2 * num_val_test:]

        dirname = os.path.dirname(list_path)
        list_name = os.path.basename(list_path).split(".")[0]
        val_path = os.path.join(dirname, list_name + "_val.lst")
        test_path = os.path.join(dirname, list_name + "_test.lst")
        train_path = os.path.join(dirname, list_name + "_train.lst")

        with open(val_path, "w") as val_file:
            val_file.writelines(list_val)
            print(f"Wrote {len(list_val)} lines to {val_path}")

        with open(test_path, "w") as test_file:
            test_file.writelines(list_test)
            print(f"Wrote {len(list_test)} lines to {test_path}")

        with open(train_path, "w") as train_file:
            train_file.writelines(list_train)
            print(f"Wrote {len(list_train)} lines to {train_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_list", help="Input list with file paths", required=True)
    parser.add_argument("--split_ratio", help="Number that determines the split between train, val, test list."
                                              "Values x < 1.0 generate a split of proportions 1-2x:x:x."
                                              "Values x > 1 represent absolute values for val and test.",
                        default=0.1)
    parser.add_argument("--seed", help="Seed for the random shuffle, a float in [0,1)", default=None)
    args = parser.parse_args()

    create_sub_lists(args.in_list, args.split_ratio, args.seed)