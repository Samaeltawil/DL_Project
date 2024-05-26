import csv
import json
import regex as re
import os


FLAGS = re.MULTILINE | re.DOTALL


# === INTERNAL FUNCTIONS ============================================================================================================
def majority_element(arr):
    count_zeros = arr.count(0)
    count_ones = arr.count(1)

    if count_zeros > count_ones:
        return 0
    elif count_ones > count_zeros:
        return 1
    else:
        return None  # No majority element


def hateful_or_not(labels):
    labels_list = labels.copy()
    for label_id in range(len(labels_list)):
        if labels_list[label_id] != 0:
            labels_list[label_id] = 1

    return majority_element(labels_list)

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = "<hashtag> {} ".format(hashtag_body.lower())
    else:
        result = " ".join(["<hashtag>"] + [re.sub(r"([A-Z])",r" \1", hashtag_body, flags=FLAGS)])
    return result

def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps>"

def tweet_preprocessing(text):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    text = re_sub(r"/"," / ")
    text = re_sub(r"<3","<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    text = re_sub(r"([A-Z]){2,}", allcaps)

    # Delete following symbols
    delete = r"[\".,-;&:]"
    text = re_sub(r"{}".format(delete), " ")

    return text.lower()

def image_is_with_text(id_image, list_img_txt_path):
    if id_image in list_img_txt_path:
        return True
    else:
        return False


# === EXTERNAL FUNCTIONS ============================================================================================================

def create_csv_labels(json_file, csv_file, img_txt_path):
    cmpt = 0
    with open(json_file, 'r') as file:
        data = json.load(file)

    with open(csv_file, 'w', newline='', encoding="utf-8") as csvfile:
        fieldnames = ['user_id', 'labels', 'hateful_label', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        list_img_txt_path = os.listdir(img_txt_path)
        # get rid of .json
        list_img_txt_path = [filename[:-5] for filename in list_img_txt_path]

        # Iterate over each user ID in the JSON file
        for user_id, user_data in data.items():
            if image_is_with_text(user_id, list_img_txt_path):
                labels = user_data.get('labels', [])
                text = user_data.get('tweet_text', [])
                text = tweet_preprocessing(text=text)
                hateful_label = hateful_or_not(labels)
                if hateful_label != 0 and hateful_label != 1:
                    continue
                # Write data to CSV file
                writer.writerow({'user_id': user_id, 'labels': labels, 'hateful_label': hateful_label, 'text': text})
                cmpt += 1
                # print(cmpt)
            else:
                continue

    print(f"Number of images with text: {cmpt}")