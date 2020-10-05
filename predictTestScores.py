import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Imports data from student-mat.csv
student_data = pd.read_csv("student-mat.csv")
student_columns = student_data.columns[0]
student_columns = student_columns.split(";")
student_data = student_data.values.tolist()

# Cleaning up data so it's in the correct format
replaced_student_data = []
for each_data in student_data:
    each_data = each_data[0]
    each_data = each_data.replace('"', '')
    each_data = each_data.split(";")
    replaced_student_data.append(each_data)
student_data = replaced_student_data


def delete(label, student_columns=student_columns, student_data=student_data):
    # Finds index of label
    label_index = student_columns.index(label)
    # Deletes index from each list
    for each_data in student_data:
        each_data.pop(label_index)
    return student_data


# Deleting columns that is not relevant to the algorithm
student_data = delete("G1")
student_data = delete("G2")
student_data = delete("nursery")


def profession_to_score(profession):
    if profession == "teacher":
        return 0
    elif profession == "health":
        return 1
    elif profession == "services":
        return 2
    elif profession == "athome":
        return 3
    return 4


def reason_to_score(reason):
    if reason == "home":
        return 0
    elif reason == "reputation":
        return 1
    elif reason == "course":
        return 2
    return 3


def guardian_to_score(guardian):
    if guardian == "mother":
        return 0
    elif guardian == "father":
        return 1
    return 2


def or_else(original, comparison):
    if original == comparison:
        return 0
    return 1


labels = []
for each_data in student_data:
    each_data[0] = or_else(each_data[0], "GP")
    each_data[1] = or_else(each_data[1], "F")
    each_data[2] = int(each_data[2])
    each_data[3] = or_else(each_data[3], "R")
    each_data[4] = or_else(each_data[4], "LE3")
    each_data[5] = or_else(each_data[5], "A")
    each_data[6] = int(each_data[6])
    each_data[7] = int(each_data[7])
    each_data[8] = profession_to_score(each_data[8])
    each_data[9] = profession_to_score(each_data[9])
    each_data[10] = reason_to_score(each_data[10])
    each_data[11] = guardian_to_score(each_data[11])
    each_data[12] = int(each_data[12])
    each_data[13] = int(each_data[13])
    each_data[14] = int(each_data[14])
    each_data[15] = or_else(each_data[15], "no")
    each_data[16] = or_else(each_data[16], "no")
    each_data[17] = or_else(each_data[17], "no")
    each_data[18] = or_else(each_data[18], "no")
    each_data[19] = or_else(each_data[19], "no")
    each_data[20] = or_else(each_data[20], "no")
    each_data[21] = or_else(each_data[21], "no")
    each_data[22] = int(each_data[22])
    each_data[23] = int(each_data[23])
    each_data[24] = int(each_data[24])
    each_data[25] = int(each_data[25])
    each_data[26] = int(each_data[26])
    each_data[27] = int(each_data[27])
    each_data[28] = int(each_data[28])
    labels.append(int(each_data[29]) * 5)
    each_data.pop(29)

# Splitting data into training and testing
x_train, x_test, y_train, y_test = train_test_split(
    student_data, labels, train_size=0.8, test_size=0.2, random_state=6)

# Using linear regression to predict test resulsts
mlr = LinearRegression()
mlr.fit(x_train, y_train)
y_predict = mlr.predict(x_test)

print(mlr.score(x_train, y_train))
print(mlr.score(x_test, y_test))
