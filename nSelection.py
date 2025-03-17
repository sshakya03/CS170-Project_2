import random
import math
import copy

def main():
    print("Welcome to Sazen Shakya's Selection Algorithm.")
    file_name = input("Type in name of the file to test: ")
    algorithm = input("Type the number of the algorithm you want to run.\n\t1) Forward Selection\n\t2) Backward Elimination\n")
    data = [list(map(float, line.split())) for line in open(file_name)]
    print(f"\nThis dataset has {len(data[0])-1} features (not including the class attribute), with {len(data)} instances.")
    print("Beginning search.")
    if algorithm == '1':
        forward_selection(data)
    else:
        backward_elimination(data)


def leave_one_out_cross_validation(data, currentSet, feature_to_add):
    copy_data = copy.deepcopy(data)
    columns_to_keep = currentSet.copy()

    if feature_to_add > 0:
        columns_to_keep.append(feature_to_add)

    for row in copy_data:
        for col_index in range(1, len(row)):
            if col_index not in columns_to_keep:
                row[col_index] = 0

    number_correctly_classified = 0
    for i in range(len(copy_data)):
        object_to_classify = copy_data[i][1:]
        label_object_to_classify = copy_data[i][0]

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_label = None
        for k in range(len(copy_data)):
            if k != i:
                distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(object_to_classify, copy_data[k][1:])))

                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_label = copy_data[k][0]

        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1
        
    accuracy = (number_correctly_classified / len(copy_data)) * 100
    return accuracy

def forward_selection(data):
    current_set_of_features = []
    best_accuracy = 0
    best_set_of_features = []

    for i in range(1, len(data[0])):
        feature_to_add_on_level = None
        best_accuracy_on_level = 0
        for k in range(1, len(data[0])):
            if k not in current_set_of_features:
                features_to_test = [k] + current_set_of_features
                accuracy = leave_one_out_cross_validation(data,features_to_test,k)
                print(f"\tUsing feature(s) {{{', '.join(map(str, features_to_test))}}} accuracy is {accuracy:.1f}%")

                if accuracy > best_accuracy_on_level:
                    best_accuracy_on_level = accuracy
                    feature_to_add_on_level = k

        if feature_to_add_on_level is not None:
            current_set_of_features.insert(0,feature_to_add_on_level)

        if best_accuracy_on_level > best_accuracy:
            best_accuracy = best_accuracy_on_level
            best_set_of_features = current_set_of_features.copy()

        if i < len(data[0])-1:
            print(f"Feature set {{{', '.join(map(str, current_set_of_features))}}} was best, accuracy is {best_accuracy_on_level:.1f}%")
    
    print(f"Finished search!! The best feature subset is {{{', '.join(map(str, best_set_of_features))}}}, which has an accuracy of {best_accuracy:.1f}%")

# Questions: Do we output the beginning feature set and accuracy? Do we output 'feature set _ was best' for the last one with only 1 thing left in the set?
def backward_elimination(data):
    current_set_of_features = list(range(1, len(data[0])))
    best_accuracy = leave_one_out_cross_validation(data,current_set_of_features,0) # 0 because no feature to add
    print(f"Feature set {{{', '.join(map(str, current_set_of_features))}}} was best, accuracy is {best_accuracy:.1f}%")

    best_set_of_features = current_set_of_features.copy()
    for i in range(1, len(data[0])-1):
        feature_to_remove_on_level = None
        best_accuracy_on_level = 0

        for k in current_set_of_features:
            features_to_test = [item for item in current_set_of_features if item != k]
            accuracy = leave_one_out_cross_validation(data, features_to_test, 0)
            print(f"\tUsing feature(s) {{{', '.join(map(str, features_to_test))}}} accuracy is {accuracy:.1f}%")

            if accuracy > best_accuracy_on_level:
                best_accuracy_on_level = accuracy
                feature_to_remove_on_level = k
        
        if feature_to_remove_on_level is not None:
            current_set_of_features.remove(feature_to_remove_on_level)
        
        if best_accuracy_on_level > best_accuracy:
            best_accuracy = best_accuracy_on_level
            best_set_of_features = current_set_of_features.copy()

        if i < len(data[0])-2:
            print(f"Feature set {{{', '.join(map(str, current_set_of_features))}}} was best, accuracy is {best_accuracy_on_level:.1f}%")
    
    print(f"Finished search!! The best feature subset is {{{', '.join(map(str, best_set_of_features))}}}, which has an accuracy of {best_accuracy:.1f}%")

if __name__ == "__main__":
    main()