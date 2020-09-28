from sklearn.model_selection import train_test_split


with open("data/kcbert/dataset_12000.txt", mode="r", encoding="utf-8") as f:
    lines = f.readlines()

train_sentences, test_sentences = train_test_split(
    lines, test_size=0.1, shuffle=True, random_state=1004
)
train_sentences, val_sentences = train_test_split(
    train_sentences, test_size=0.1, shuffle=True, random_state=1004
)


def write_file(data_path, sentences):
    with open(data_path, mode="w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s)


write_file("data/kcbert/train_data_12000.txt", train_sentences)
write_file("data/kcbert/val_data_12000.txt", val_sentences)
write_file("data/kcbert/test_data_12000.txt", test_sentences)
